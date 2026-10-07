"""SM107 NVFP4 MegaMoE: per-call ``fc1_norm_const`` / ``fc2_alpha`` plumbing and why it matters.

Single Rubin GPU (``MEGA_NO_DIST=1``)::

    MEGA_NO_DIST=1 CUTE_DSL_ARCH=sm_107a python -m pytest tests/moe_ep/test_sm107_nvfp4_norm_const.py -m arch_rubin

1. Per-expert ``fc1_norm_const`` and ``fc2_alpha`` passed through ``MoEEpTensors`` are staged
   and used by the kernel (shim oracle over the same staged bytes and scalars).
2. With small FC1 outputs (DeepSeek-V4.1 layers reach amax ~0.04), the default unit norm
   const underflows the E4M3 block scales of the FC1->FC2 requant. The calibrated
   ``fc1_norm_const = 1 / w2.input_scale`` (``sm107_nvfp4_checkpoint_scalars``) restores the
   precision: error vs an fp64 MoE without the requant drops by more than 1.5x.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.arch_rubin

H, I, E, TOP_K, T = 1024, 256, 8, 4, 96


def _skip_unless_rubin():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires an SM107 GPU")
    if os.environ.get("MEGA_NO_DIST") != "1":
        pytest.skip("single-rank test: export MEGA_NO_DIST=1")


def _layer(transformed=None, weights=None, input_norm_const=1.0):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    cfg = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=I, top_k=TOP_K, input_norm_const=input_norm_const
    )
    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=E, max_tokens_per_rank=128, token_hidden_size=H
        ),
        weights=weights,
        backend=MegaConfig(megakernel=cfg, transformed_weights=transformed),
    )


def _problem(x_scale, w_scale=8.0):
    # w_scale keeps the weights' own NVFP4 block scales in the E4M3 normal range
    g = torch.Generator(device="cuda").manual_seed(7)
    w13 = (
        torch.randn(E, 2 * I, H, device="cuda", generator=g) * (w_scale * H**-0.5)
    ).bfloat16()
    w2 = (
        torch.randn(E, H, I, device="cuda", generator=g) * (w_scale * I**-0.5)
    ).bfloat16()
    x = (torch.randn(T, H, device="cuda", generator=g) * x_scale).bfloat16()
    ids = torch.rand(T, E, device="cuda", generator=g).argsort(1)[:, :TOP_K].int()
    w = (torch.rand(T, TOP_K, device="cuda", generator=g) + 0.5).float()
    return w13, w2, x, ids, w


def _fp64_moe(x, w13, w2, ids, w):
    """MoE in fp64 from the canonical bf16 weights, no activation / intermediate quant."""
    xd = x.double()
    y = torch.zeros(T, H, dtype=torch.float64, device="cuda")
    for e in range(E):
        r, k = (ids.long() == e).nonzero(as_tuple=True)
        if r.numel() == 0:
            continue
        h = xd[r] @ w13[e].double().T
        act = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        y.index_add_(0, r, (act @ w2[e].double().T) * w[r, k].double().unsqueeze(1))
    return y


def test_per_call_norm_const_and_alpha_reach_the_kernel():
    _skip_unless_rubin()
    import flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe as pkg
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack

    w13, w2, x, ids, w = _problem(0.125)
    layer = _layer(weights=MoEWeightPack(w13=w13, w2=w2))
    try:
        g = torch.Generator(device="cuda").manual_seed(3)
        # not powers of two: a 2^k norm const only shifts the E4M3 exponent and leaves
        # the requantized FC2 input (and so the output) bit-identical
        nc = (torch.rand(E, device="cuda", generator=g) * 11.0 + 0.6).float()
        a1 = (torch.rand(E, device="cuda", generator=g) + 0.5).float()
        a2 = (1.0 / nc).contiguous()
        y = layer.forward(
            MoEEpTensors(x, ids, w, fc1_alpha=a1, fc2_alpha=a2, fc1_norm_const=nc)
        )
        ws = layer._ensure_workspace()
        # bound per-call scalars (default) or copied ones (FLASHINFER_SM107_MEGA_COPY_SCALARS=1)
        assert torch.equal(ws._scalar_tensor("fc1_norm_const"), nc)
        assert torch.equal(ws._scalar_tensor("fc2_alpha"), a2)
        idx, exp = pkg.sampled_reference(ws, *layer._transformed, T)
        assert pkg.output_error(y, idx, exp) < 0.06
        # a different (non power-of-two) norm const changes the requant grid -> different output
        y1 = layer.forward(MoEEpTensors(x, ids, w, fc1_alpha=a1))
        assert not torch.equal(y, y1)
    finally:
        layer.destroy()


def test_calibrated_norm_const_fixes_small_activation_underflow():
    _skip_unless_rubin()
    from flashinfer.moe_ep import (
        MoEEpTensors,
        MoEWeightPack,
        sm107_nvfp4_checkpoint_scalars,
    )

    # x ~ N(0, 0.005^2): FC1 outputs ~0.04, SwiGLU outputs ~1e-3 -> unit-nc block scales (amax/6)
    # fall below the E4M3 normal range (2^-6) and the FC2 input loses its precision
    w13, w2, x, ids, w = _problem(0.005)
    ref = _fp64_moe(x, w13, w2, ids, w)
    x_gs = 448.0 * 6.0 / float(x.float().abs().amax())  # calibrated input norm const
    layer = _layer(weights=MoEWeightPack(w13=w13, w2=w2), input_norm_const=x_gs)
    try:
        # calibration as modelopt does it: w2.input_scale = amax(FC2 input) / (6 * 448)
        act_amax = 0.0
        for e in range(E):
            h = x.double() @ w13[e].double().T
            act_amax = max(
                act_amax,
                float((torch.nn.functional.silu(h[:, :I]) * h[:, I:]).abs().amax()),
            )
        w2_input_scale = torch.full((E,), act_amax / (6.0 * 448.0), device="cuda")
        ones = torch.ones(E, device="cuda")
        a1, a2, nc = sm107_nvfp4_checkpoint_scalars(
            ones, ones, w2_input_scale, input_norm_const=x_gs
        )
        assert torch.allclose(nc, 1.0 / w2_input_scale) and torch.allclose(
            a2, w2_input_scale
        )
        y_cal = layer.forward(
            MoEEpTensors(x, ids, w, fc1_alpha=a1, fc2_alpha=a2, fc1_norm_const=nc)
        )
        a1u, a2u, ncu = sm107_nvfp4_checkpoint_scalars(
            ones, ones, None, input_norm_const=x_gs
        )
        y_unit = layer.forward(
            MoEEpTensors(x, ids, w, fc1_alpha=a1u, fc2_alpha=a2u, fc1_norm_const=ncu)
        )
        err = lambda y: float((y.double() - ref).norm() / ref.norm())  # noqa: E731
        e_cal, e_unit = err(y_cal), err(y_unit)
        print(
            f"rel_l2 vs fp64 bf16-weight MoE: calibrated nc {e_cal:.4f}, unit nc {e_unit:.4f}"
        )
        # e_cal includes the NVFP4 weight, input and FC2-input quantization (~0.2 floor here)
        assert e_cal < 0.35, e_cal
        assert e_unit > 1.5 * e_cal and e_unit - e_cal > 0.1, (e_unit, e_cal)
    finally:
        layer.destroy()
