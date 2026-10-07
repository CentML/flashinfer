# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Clear the pre-reduced combine slots of invalid routes (graph-safe, cost ~ live routes).

The SM107 kernels never write the combine slot of an invalid route (``-1`` or an expert
id outside ``[0, E)``), but the separate ``TopkReduce`` sums every ``(token, k)`` slot, so
such slots must be cleared before each launch. ``masked_fill_`` over the whole
``(capacity, topk, row)`` buffer reads and rewrites it every launch: ~2 GB of traffic per
launch at capacity 16384 / H 5120 / top-6 BF16, i.e. ~25% of the MegaMoE step time.
This kernel instead reads the ``num_tokens * topk`` route ids and writes only the rows of
invalid routes among the staged tokens (rows at or beyond the staged count are never
consumed: the caller reads ``output[:num_tokens]``).

``FLASHINFER_SM107_MEGA_TORCH_CLEAR=1`` restores the full-buffer ``masked_fill_`` (and the
full-capacity TopkReduce / in-kernel-reduce zeroing, see ``block_scaled.launch``).
"""

from __future__ import annotations

import os

import torch

_KERNEL = None


def use_torch_clear() -> bool:
    return os.environ.get("FLASHINFER_SM107_MEGA_TORCH_CLEAR", "0") == "1"


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _clear_invalid_route_slots(
        ids_ptr,
        slots_ptr,
        num_slots,
        num_experts,
        row_words,
        fill_word,
        SLOTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # SLOTS route slots per program; the common all-valid block exits after one
        # vector load of its ids.
        sidx = tl.program_id(0) * SLOTS + tl.arange(0, SLOTS)
        in_range = sidx < num_slots
        e = tl.load(ids_ptr + sidx, mask=in_range, other=0)
        invalid = in_range & ((e < 0) | (e >= num_experts))
        if tl.sum(invalid.to(tl.int32), axis=0) > 0:
            base = sidx.to(tl.int64) * row_words
            value = tl.full([SLOTS, BLOCK], 0, dtype=tl.int32) + fill_word
            for start in range(0, row_words, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                mask = invalid[:, None] & (offs[None, :] < row_words)
                tl.store(slots_ptr + base[:, None] + offs[None, :], value, mask=mask)

    _KERNEL = _clear_invalid_route_slots
    return _KERNEL


def clear_invalid_route_slots(
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    slots: torch.Tensor,
    fill_byte: int = 0,
) -> None:
    """Fill ``slots[t, k, :]`` with ``fill_byte`` for invalid routes of tokens ``t < num_tokens``.

    ``topk_idx`` is the contiguous int32 ``(capacity, topk)`` routing plane and ``slots`` a
    contiguous ``(capacity, topk, row_bytes)`` byte view (``row_bytes % 4 == 0``).
    """
    if num_tokens <= 0:
        return
    if not slots.is_contiguous() or not topk_idx.is_contiguous():
        raise ValueError("clear_invalid_route_slots needs contiguous tensors")
    raw = slots.view(
        torch.uint8
    )  # (capacity, topk, row_bytes) whatever the element type
    capacity, topk, row_bytes = raw.shape
    if row_bytes % 4:
        raise ValueError("clear_invalid_route_slots needs 4-byte-multiple rows")
    words = raw.view(torch.int32)
    fill = int(fill_byte) & 0xFF
    fill_word = fill | (fill << 8) | (fill << 16) | (fill << 24)
    if fill_word >= 1 << 31:
        fill_word -= 1 << 32
    row_words = row_bytes // 4
    num_slots = int(num_tokens) * topk
    slots_per_prog = 32
    grid = ((num_slots + slots_per_prog - 1) // slots_per_prog,)
    _get_kernel()[grid](
        topk_idx,
        words,
        num_slots,
        int(num_experts),
        row_words,
        fill_word,
        SLOTS=slots_per_prog,
        BLOCK=256,
    )


__all__ = ["clear_invalid_route_slots", "use_torch_clear"]
