"""NVFP4 MegaMoE staging that quantizes straight into the workspace (CUDA, SM100+ CuTe DSL).

``nvfp4_quantize_cute_dsl(out=(fp4, sf))`` writes caller-owned buffers and must produce the same
bytes as the allocating call; the SM107 NVFP4 staging uses it to quantize the BF16 activations
straight into the workspace ``x`` / ``x_sf`` (``FLASHINFER_SM107_MEGA_DIRECT_STAGE``, default on)
and must stage the same bytes as quantize-then-copy (``FLASHINFER_SM107_MEGA_DIRECT_STAGE=0``).
Token counts cover both quantizer paths (vectorized below the TMA crossover, TMA above it).
"""

import pytest
import torch


def _skip_unless_supported():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 CuTe-DSL quantization needs compute capability >= 10")
    pytest.importorskip("cutlass")


TOKENS = [1, 37, 1260, 2304, 4100]


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("hidden", [4096, 5120])
def test_nvfp4_quantize_cute_dsl_out_matches_allocating_call(tokens, hidden):
    _skip_unless_supported()
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        SF_LAYOUT_LINEAR,
        nvfp4_quantize_cute_dsl,
    )

    torch.manual_seed(tokens)
    x = (torch.randn(tokens, hidden, device="cuda") * 3).to(torch.bfloat16)
    q, sf = nvfp4_quantize_cute_dsl(
        x, 0.7, sf_layout=SF_LAYOUT_LINEAR, nvfp4_4over6=None
    )
    rows = ((tokens + 15) // 16) * 16 + 16
    q_out = torch.full((tokens, hidden // 2), 0xA5, dtype=torch.uint8, device="cuda")
    sf_out = torch.full((rows, hidden // 16), 0x5A, dtype=torch.uint8, device="cuda")
    q2, sf2 = nvfp4_quantize_cute_dsl(
        x,
        0.7,
        sf_layout=SF_LAYOUT_LINEAR,
        nvfp4_4over6=None,
        out=(q_out, sf_out.view(-1)),
    )
    assert q2.data_ptr() == q_out.data_ptr() and sf2.data_ptr() == sf_out.data_ptr()
    assert torch.equal(q_out, q.view(torch.uint8))
    n = sf.numel()
    assert torch.equal(sf_out.view(-1)[:n], sf.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("tokens", TOKENS)
def test_direct_staging_equals_quantize_then_copy(monkeypatch, tokens):
    _skip_unless_supported()
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )

    hidden, topk, capacity = 5120, 6, 4224
    torch.manual_seed(tokens + 1)
    x = (torch.randn(tokens, hidden, device="cuda") * 2).to(torch.bfloat16)
    ids = torch.randint(0, 384, (tokens, topk), device="cuda", dtype=torch.int32)
    w = torch.rand(tokens, topk, device="cuda")
    staged = []
    for direct in ("1", "0"):
        monkeypatch.setenv("FLASHINFER_SM107_MEGA_DIRECT_STAGE", direct)
        xq = torch.full((capacity, hidden // 2), 0x11, dtype=torch.uint8, device="cuda")
        xsf = torch.full(
            (capacity, hidden // 16), 0x22, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        idx = torch.empty(capacity, topk, dtype=torch.int32, device="cuda")
        wts = torch.empty(capacity, topk, dtype=torch.float32, device="cuda")
        assert (
            stage_mega_moe_inputs(x, w, ids, xq, xsf, idx, wts, input_norm_const=0.5)
            == tokens
        )
        staged.append((xq, xsf.view(torch.uint8), idx, wts))
    (q1, s1, i1, w1), (q0, s0, i0, w0) = staged
    assert torch.equal(q1[:tokens], q0[:tokens])
    assert torch.equal(s1[:tokens], s0[:tokens])
    assert torch.equal(i1, i0) and torch.equal(w1[:tokens], w0[:tokens])
