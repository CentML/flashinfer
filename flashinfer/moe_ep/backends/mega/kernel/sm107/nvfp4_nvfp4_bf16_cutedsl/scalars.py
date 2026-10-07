"""Per-expert NVFP4 scalars for the SM107 MegaMoE kernel from a modelopt NVFP4 checkpoint.

The SM107 NVFP4 kernel takes three per-local-expert fp32 vectors (``MoEEpTensors`` or the
config): ``fc1_alpha`` (FC1 accumulator scale), ``fc1_norm_const`` (normalisation of the
in-kernel FC1-output -> FC2-input NVFP4 requant: block scale ``e4m3(amax/6 * nc)``) and
``fc2_alpha`` (FC2 accumulator scale). Leaving ``fc1_norm_const`` at its default of 1 makes the
FC1-output block scales ``amax/6``; for DeepSeek-V4.1 layers whose FC1 outputs are small
(amax ~0.04) these underflow the E4M3 normal range and the FC2 input loses most of its
precision (72% vs 8.8% wire error measured on VR200, WP5a). The checkpoint's calibrated
``w2.input_scale`` gives the right normalisation, exactly as TensorRT-LLM does
(``_get_fc2_alpha_input_scale``): ``fc1_norm_const = 1 / w2.input_scale`` and
``fc2_alpha = w2.weight_scale_2 * w2.input_scale`` (= ``w2_s2 / fc1_norm_const``).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

ScaleLike = Union[torch.Tensor, float]


def _vec(value: ScaleLike, n: int, device) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(n)
    if t.numel() != n:
        raise ValueError(f"expected 1 or {n} values, got {t.numel()}")
    return t.contiguous()


def sm107_nvfp4_checkpoint_scalars(
    w13_weight_scale_2: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w2_input_scale: Optional[ScaleLike],
    *,
    input_norm_const: ScaleLike = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(fc1_alpha, fc2_alpha, fc1_norm_const)`` as contiguous fp32 ``[E_local]``.

    Args:
        w13_weight_scale_2: ``[E_local]`` or ``[E_local, 2]`` (gate, up) per-tensor FC1 weight
            scales. With two columns they must be equal (fold an unequal ratio into the up
            half's E4M3 block scales first, as vLLM's ``nvfp4_prequant_pack_and_alphas`` does).
        w2_weight_scale_2: ``[E_local]`` per-tensor FC2 weight scales.
        w2_input_scale: ``[E_local]`` (or a scalar) calibrated FC2 input scales
            (``amax / (6 * 448)``). ``None`` keeps the legacy unit norm const (not recommended).
        input_norm_const: the FC1-input normalisation the activations were quantized with
            (``MegaConfig(quantize_input=True)`` staging uses the config's
            ``input_norm_const``; a caller-side quantizer uses its global scale, e.g.
            ``1 / w13.input_scale``). ``fc1_alpha`` divides it back out.
    """
    s13 = w13_weight_scale_2.float()
    if s13.dim() == 2:
        if s13.shape[1] != 2:
            raise ValueError("w13_weight_scale_2 must be [E] or [E, 2]")
        if not torch.equal(s13[:, 0], s13[:, 1]):
            raise ValueError(
                "gate and up weight_scale_2 differ; fold the ratio into the up block scales "
                "before calling (the kernel has one fc1_alpha per expert)"
            )
        s13 = s13[:, 0]
    n = s13.numel()
    dev = s13.device
    in_nc = _vec(input_norm_const, n, dev)
    fc1_alpha = (s13.reshape(-1) / in_nc).contiguous()
    s2 = _vec(w2_weight_scale_2, n, dev)
    if w2_input_scale is None:
        nc = torch.ones(n, dtype=torch.float32, device=dev)
    else:
        in2 = _vec(w2_input_scale, n, dev)
        if bool((in2 <= 0).any()):
            raise ValueError("w2_input_scale must be positive")
        nc = (1.0 / in2).contiguous()
    fc2_alpha = (s2 / nc).contiguous()
    return fc1_alpha, fc2_alpha, nc


__all__ = ["sm107_nvfp4_checkpoint_scalars"]
