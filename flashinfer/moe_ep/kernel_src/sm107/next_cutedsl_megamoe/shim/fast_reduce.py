# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Masked top-k combine reduce with an optional fused addend (FlashInfer-owned shim).

Replaces the separate-reduce tail of the SM107 MegaMoE launch, i.e. the vendored
``TopkReduce`` (BF16 combine) plus the per-launch invalid-slot clear
(``clear_invalid.clear_invalid_route_slots``), with ONE Triton launch:

    y[t, :] = bf16( sum_k  valid(t, k) * score(t, k) * pre[t, k, :] )          (fp32 sum)
    y[t, :] = bf16( float(y[t, :]) + float(addend[t, :]) )                       (optional)

* ``valid(t, k)``: the staged route id is in ``[0, E)``. Invalid routes never get a
  combine slot written by the kernel; instead of zeroing those slots before every
  launch, their loads are masked (``other=0``), so the sum is bit-identical to
  TopkReduce over cleared slots (``+0`` terms, same fp32 order k = 0..K-1, same
  first-term initialisation, ``fma(v, s, acc)`` when scores are applied here).
* ``addend`` (e.g. the shared-expert MLP output): the second rounding reproduces
  ``y = TopkReduce(...); y += addend`` (BF16 tensor add = fp32 add + RN) bit for bit,
  without the separate elementwise kernel and its extra read / write of ``y``.

``FLASHINFER_SM107_MEGA_FAST_REDUCE=0`` restores the vendored TopkReduce + clear.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_KERNEL = None


def use_fast_reduce() -> bool:
    return os.environ.get("FLASHINFER_SM107_MEGA_FAST_REDUCE", "1") != "0"


def _block_config():
    """(BLOCK_T, BLOCK_H, num_warps, masked_loads); overridable for tuning sweeps:
    FLASHINFER_SM107_MEGA_FAST_REDUCE_BLOCK="bt,bh,nw[,masked]"."""
    spec = os.environ.get("FLASHINFER_SM107_MEGA_FAST_REDUCE_BLOCK", "")
    if spec:
        vals = [int(v) for v in spec.split(",")]
        bt, bh, nw = vals[:3]
        masked = bool(vals[3]) if len(vals) > 3 else False
        return bt, bh, nw, masked
    return 1, 1024, 4, False


def _use_pdl(device) -> bool:
    """PDL launch (FLASHINFER_SM107_MEGA_FAST_REDUCE_PDL, default on for SM90+)."""
    if os.environ.get("FLASHINFER_SM107_MEGA_FAST_REDUCE_PDL", "1") == "0":
        return False
    return torch.cuda.get_device_capability(device)[0] >= 9


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _masked_topk_reduce(
        pre_ptr,
        ids_ptr,
        score_ptr,
        add_ptr,
        out_ptr,
        num_tokens,
        num_experts,
        pre_stride_t,
        pre_stride_k,
        out_stride_t,
        add_stride_t,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_SCORE: tl.constexpr,
        HAS_ADD: tl.constexpr,
        MASKED_LOADS: tl.constexpr,
        LAUNCH_PDL: tl.constexpr,
    ):
        if LAUNCH_PDL:
            # Programmatic dependent launch: the launch overlaps the megakernel's
            # tail; every read below waits for the full primary grid.
            tl.extra.cuda.gdc_wait()
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        tmask = t < num_tokens
        hmask = h < H
        t64 = t.to(tl.int64)
        m2 = tmask[:, None] & hmask[None, :]
        acc = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            e = tl.load(ids_ptr + t64 * TOPK + k, mask=tmask, other=-1)
            valid = tmask & (e >= 0) & (e < num_experts)
            ptrs = pre_ptr + t64[:, None] * pre_stride_t + k * pre_stride_k + h[None, :]
            if MASKED_LOADS:
                v = tl.load(
                    ptrs,
                    mask=valid[:, None] & hmask[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                # Load every live slot (no dependency on the id load), then select:
                # an invalid route's stale slot (any bits, NaN included) becomes +0.
                v = tl.load(ptrs, mask=m2, other=0.0, eviction_policy="evict_first")
                v = tl.where(valid[:, None], v.to(tl.float32), 0.0)
            if HAS_SCORE:
                s = tl.load(score_ptr + t64 * TOPK + k, mask=tmask, other=0.0)
                if k == 0:
                    acc = v * s[:, None]
                else:
                    acc = tl.fma(v, s[:, None], acc)
            else:
                if k == 0:
                    acc = v
                else:
                    acc = acc + v
        y = acc.to(tl.bfloat16)
        if HAS_ADD:
            a = tl.load(
                add_ptr + t64[:, None] * add_stride_t + h[None, :], mask=m2, other=0.0
            )
            y = (y.to(tl.float32) + a.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + t64[:, None] * out_stride_t + h[None, :], y, mask=m2)

    _KERNEL = _masked_topk_reduce
    return _KERNEL


def masked_topk_reduce(
    pre_reduced: torch.Tensor,
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    out: torch.Tensor,
    *,
    scores: Optional[torch.Tensor] = None,
    addend: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[:num_tokens] = reduce(pre_reduced[:num_tokens]) (+ addend[:num_tokens])``.

    ``pre_reduced``: BF16 ``(capacity, topk, >= hidden)`` combine slots (row stride may be
    padded); ``topk_idx``: contiguous int32 ``(capacity, topk)``; ``scores``: optional fp32
    ``(capacity, topk)`` router weights (only when they were not applied at FC1);
    ``out`` / ``addend``: BF16 ``(>= num_tokens, hidden)`` with unit inner stride.
    Graph-safe (host ints only)."""
    if num_tokens <= 0:
        return out
    hidden = out.shape[1]
    topk = topk_idx.shape[1]
    if pre_reduced.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("masked_topk_reduce needs BF16 combine slots and output")
    if pre_reduced.stride(2) != 1 or out.stride(1) != 1 or not topk_idx.is_contiguous():
        raise ValueError("masked_topk_reduce needs unit inner strides / contiguous ids")
    if pre_reduced.shape[1] != topk or pre_reduced.shape[2] < hidden:
        raise ValueError(
            f"combine slots {tuple(pre_reduced.shape)} do not match topk {topk} / hidden {hidden}"
        )
    if addend is not None:
        if (
            addend.dtype != torch.bfloat16
            or addend.stride(-1) != 1
            or addend.shape[-1] != hidden
        ):
            raise ValueError(
                "addend must be BF16 (tokens, hidden) with unit inner stride"
            )
        addend = addend.reshape(-1, hidden)
        if addend.shape[0] < num_tokens:
            raise ValueError("addend has fewer rows than num_tokens")
    if scores is not None and (
        scores.dtype != torch.float32 or not scores.is_contiguous()
    ):
        raise ValueError("scores must be contiguous fp32")
    import triton

    bt, bh, nw, masked = _block_config()
    pdl = _use_pdl(out.device)
    grid = (triton.cdiv(int(num_tokens), bt), triton.cdiv(hidden, bh))
    _get_kernel()[grid](
        pre_reduced,
        topk_idx,
        scores if scores is not None else topk_idx,
        addend if addend is not None else out,
        out,
        int(num_tokens),
        int(num_experts),
        pre_reduced.stride(0),
        pre_reduced.stride(1),
        out.stride(0),
        addend.stride(0) if addend is not None else 0,
        H=hidden,
        TOPK=topk,
        BLOCK_T=bt,
        BLOCK_H=bh,
        HAS_SCORE=scores is not None,
        HAS_ADD=addend is not None,
        MASKED_LOADS=masked,
        LAUNCH_PDL=pdl,
        num_warps=nw,
        **({"launch_pdl": True} if pdl else {}),
    )
    return out


__all__ = ["masked_topk_reduce", "use_fast_reduce"]
