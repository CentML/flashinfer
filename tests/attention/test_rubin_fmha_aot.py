# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Numerical AOT-only Rubin integration tests.

ViT uses the published artifact pin. Paged tests require locally rebuilt HND
exports (FLASHINFER_DSL_FMHA_LOCAL_DIR) until those are published. Missing
artifacts fail, never JIT. Tolerances are FP8/FP4 smoke gates, not MLPerf accuracy.
"""

import math
import os

import pytest
import torch

from flashinfer.attention.cute_dsl import (
    cute_dsl_fmha_paged_prefill,
    cute_dsl_fmha_vit,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Requires Rubin SM107",
)


def _indptr(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )


def _fp8(shape):
    return (torch.randn(shape, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def _reference(q, k, v, qlens, klens, scale, vscale, causal):
    outputs, iq, ik = [], 0, 0
    for nq, nk in zip(qlens, klens, strict=True):
        qr = q[iq : iq + nq].float().transpose(0, 1)
        kr = k[ik : ik + nk].float().transpose(0, 1)
        vr = v[ik : ik + nk].float().transpose(0, 1)
        repeat = qr.shape[0] // kr.shape[0]
        kr, vr = kr.repeat_interleave(repeat, 0), vr.repeat_interleave(repeat, 0)
        score = (qr @ kr.transpose(-1, -2)) * scale
        if causal:
            invalid = (
                torch.arange(nk, device="cuda")[None, :]
                > torch.arange(nq, device="cuda")[:, None] + nk - nq
            )
            score.masked_fill_(invalid, -float("inf"))
        outputs.append((score.softmax(-1) @ (vr * vscale)).transpose(0, 1))
        iq, ik = iq + nq, ik + nk
    return torch.cat(outputs)


def _graph_replay(run, *outputs):
    run()
    eager = [output.clone() for output in outputs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for output in outputs:
        output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for output, expected in zip(outputs, eager, strict=True):
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("padded_graph", [False, True])
def test_rubin_vit_d72_d80_aot(padded_graph):
    torch.manual_seed(199)
    lengths = [64, 0, 32, 0] if padded_graph else [64, 129]
    active = sum(lengths)
    capacity = 128 if padded_graph else active
    tensors = []
    for _ in range(3):
        padded = torch.zeros((capacity, 16, 80), device="cuda")
        padded[..., :72] = torch.randn((capacity, 16, 72), device="cuda") * 0.5
        tensors.append(padded.to(torch.float8_e4m3fn))
    q, k, v = tensors
    o = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
    o[active:].fill_(-17)
    indptr = _indptr(lengths)

    def run():
        cute_dsl_fmha_vit(
            q, k, v, o, indptr, max(lengths), scale_q=0.5, scale_k=0.25, scale_v=2.0
        )

    # Padding channels are not defined by the kernel and excluded from checks.
    _graph_replay(run, o[:active, ..., :72])
    ref = _reference(
        q[..., :72],
        k[..., :72],
        v[..., :72],
        lengths,
        lengths,
        0.125 / math.sqrt(72),
        2.0,
        False,
    )
    torch.testing.assert_close(o[:active, ..., :72].float(), ref, rtol=0.05, atol=0.05)
    assert torch.all(o[active:] == -17)


@pytest.mark.parametrize(
    "fp4,output_store", [(False, "direct"), (True, "direct"), (True, "ragged_tma")]
)
@pytest.mark.parametrize("cache_layout", ["combined_hnd", "split_d"])
@pytest.mark.parametrize("sf_offset", [0, 3, 127])
def test_rubin_paged_gqa_aot(fp4, output_store, cache_layout, sf_offset):
    if not os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR"):
        pytest.skip("Requires rebuilt HND / NVFP4 exporter artifacts")
    torch.manual_seed(199)
    qlens, klens, page = [64, 129], [95, 177], 16
    total, hq, hk, d = sum(qlens), 64, 4, 128
    q, k, v = _fp8((total, hq, d)), _fp8((sum(klens), hk, d)), _fp8((sum(klens), hk, d))
    counts = [math.ceil(n / page) for n in klens]
    pages = sum(counts)
    if cache_layout == "combined_hnd":
        pool = torch.zeros((pages, 2, hk, page, d), device="cuda").to(q.dtype)
        kc, vc = pool[:, 0], pool[:, 1]
    else:
        pool = torch.zeros((pages, hk, page, 2 * d), device="cuda").to(q.dtype)
        kc, vc = pool.split(d, dim=-1)
    table = torch.zeros((2, max(counts)), dtype=torch.int32, device="cuda")
    order, ip, it = torch.randperm(pages).tolist(), 0, 0
    for batch, nk in enumerate(klens):
        for p in range(counts[batch]):
            physical, n = order[ip + p], min(page, nk - p * page)
            table[batch, p] = physical
            kc[physical, :, :n].copy_(
                k[it + p * page : it + p * page + n].transpose(0, 1)
            )
            vc[physical, :, :n].copy_(
                v[it + p * page : it + p * page + n].transpose(0, 1)
            )
        ip, it = ip + counts[batch], it + nk
    o = torch.empty(
        (total, hq, d // 2 if fp4 else d),
        dtype=torch.uint8 if fp4 else torch.bfloat16,
        device="cuda",
    )
    rows, cols = math.ceil((sf_offset + total) / 128) * 128, hq * d // 16
    sf = torch.zeros((rows, cols), dtype=q.dtype, device="cuda")
    cq, ck = _indptr(qlens), _indptr(klens)
    lengths = torch.tensor(klens, dtype=torch.int32, device="cuda")

    def run():
        cute_dsl_fmha_paged_prefill(
            q,
            kc,
            vc,
            o,
            cq,
            ck,
            table,
            lengths,
            max(qlens),
            max(klens),
            scale_q=0.5,
            scale_k=0.25,
            scale_v=2.0,
            output_block_scale=sf if fp4 else None,
            o_sf_scale=64.0 if fp4 else None,
            o_sf_start_index=sf_offset if fp4 else 0,
            output_store=output_store,
        )

    if fp4:
        _graph_replay(run, o, sf)
    else:
        _graph_replay(run, o)
    ref = _reference(q, k, v, qlens, klens, 0.125 / math.sqrt(d), 2.0, True)
    if not fp4:
        torch.testing.assert_close(o.float(), ref, atol=0.05, rtol=0.05)
        return
    # Decode E2M1 and independently address the full swizzled scale buffer.
    payload = o.reshape(total, -1)
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(total, -1).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    r = torch.arange(rows, device="cuda")[:, None]
    c = torch.arange(cols, device="cuda")[None, :]
    index = (
        (r // 128) * (128 * cols)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )
    scales = sf.flatten().float()[index]
    assert torch.count_nonzero(scales[:sf_offset]) == 0
    dequant = (
        values[codes]
        * scales[sf_offset : sf_offset + total].repeat_interleave(16, dim=1)
        / 64.0
    )
    assert torch.isfinite(dequant).all()
    relative_l2 = (dequant - ref.reshape(total, -1)).norm() / ref.norm()
    assert relative_l2 < 0.2, relative_l2.item()
