# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Numerical AOT-only Rubin integration tests.

Every export loads from the pinned cute-dsl FMHA collection
(ArtifactPath.DSL_FMHA), or from FLASHINFER_DSL_FMHA_LOCAL_DIR when it is set.
Missing artifacts fail; they never JIT and never skip. Tolerances are FP8/FP4
smoke gates, not MLPerf accuracy.
"""

import math

import pytest
import torch

from flashinfer.attention.cute_dsl import (
    RUBIN_GQA_TUNED_POLICY,
    cute_dsl_fmha_paged_prefill,
    cute_dsl_fmha_vit,
    cute_dsl_fmha_vit_work_count,
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


@pytest.mark.parametrize("mma_rowsum", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_rubin_vit_d72_persistent_aot(dynamic, mma_rowsum):
    torch.manual_seed(199)
    # Partial 256-row tiles, empty graph-padding slots and an unused tail.
    lengths = [300, 0, 1024, 17, 0, 0]
    active = sum(lengths)
    capacity = active + 64
    tensors = []
    for _ in range(3):
        padded = torch.zeros((capacity, 16, 80), device="cuda")
        padded[:active, :, :72] = torch.randn((active, 16, 72), device="cuda") * 0.5
        tensors.append(padded.to(torch.float8_e4m3fn))
    q, k, v = tensors
    o = torch.zeros(q.shape, dtype=torch.bfloat16, device="cuda")
    o[active:].fill_(-17)
    indptr = _indptr(lengths)
    counter = None
    if dynamic:
        work = cute_dsl_fmha_vit_work_count(lengths, 16)
        counter = torch.tensor([0, work], dtype=torch.int32, device="cuda")

    def run():
        # Encoder layers launch back to back with one shared counter, and
        # max_seqlen is the graph capacity rather than the longest sequence.
        for _ in range(2):
            cute_dsl_fmha_vit(
                q,
                k,
                v,
                o,
                indptr,
                capacity,
                scale_q=0.5,
                scale_k=0.25,
                scale_v=2.0,
                persistent=True,
                work_counter=counter,
                mma_rowsum=mma_rowsum,
            )

    _graph_replay(run, o[:active, ..., :72])
    if dynamic:
        # The last Work-ID claimant resets the queue for the next launch.
        assert counter.tolist() == [0, work]
    ref = _reference(
        q[:active, ..., :72],
        k[:active, ..., :72],
        v[:active, ..., :72],
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


def _paged_inputs(qlens, klens, cache_layout, page=16, hq=64, hk=4, d=128):
    total = sum(qlens)
    q, k, v = _fp8((total, hq, d)), _fp8((sum(klens), hk, d)), _fp8((sum(klens), hk, d))
    counts = [math.ceil(n / page) for n in klens]
    pages = sum(counts)
    if cache_layout == "combined_hnd":
        pool = torch.zeros((pages, 2, hk, page, d), device="cuda").to(q.dtype)
        kc, vc = pool[:, 0], pool[:, 1]
    else:  # vLLM split-D: [pages, Hkv, page, 2 * D]
        pool = torch.zeros((pages, hk, page, 2 * d), device="cuda").to(q.dtype)
        kc, vc = pool.split(d, dim=-1)
    table = torch.zeros((len(klens), max(counts)), dtype=torch.int32, device="cuda")
    order, ip, it = torch.randperm(pages).tolist(), 0, 0
    for batch, nk in enumerate(klens):
        for p in range(counts[batch]):
            physical, n = order[ip + p], min(page, nk - p * page)
            table[batch, p] = physical
            src = slice(it + p * page, it + p * page + n)
            kc[physical, :, :n].copy_(k[src].transpose(0, 1))
            vc[physical, :, :n].copy_(v[src].transpose(0, 1))
        ip, it = ip + counts[batch], it + nk
    return q, k, v, kc, vc, table


def _scale_rows(sf, rows, cols):
    """Swizzled Layout128x4 scale buffer -> [rows, cols] in logical order."""
    r = torch.arange(rows, device="cuda")[:, None]
    c = torch.arange(cols, device="cuda")[None, :]
    index = (
        (r // 128) * (128 * cols)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )
    return sf.flatten().view(torch.uint8)[index]


_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]

# Q/KV lengths per case; every layout of the tuned policy, plus a mixed
# decode + chunked-prefill batch (5 single-token decodes, 3 x 512 chunks).
_TUNED_CASES = {
    "uniform": ([192] * 3, [192, 300, 512], "uniform", "tma"),
    "packed_ragged": ([64, 129, 300], [95, 177, 600], "packed", "ragged_tma"),
    "packed_dense": ([256, 512, 77], [256, 700, 77], "packed", "tma"),
    "packed_direct": ([64, 129, 300], [95, 177, 600], "packed", "direct"),
    "mixed_ragged": (
        [1] * 5 + [512] * 3,
        [2048, 777, 1500, 33, 4096, 512, 1536, 3512],
        "packed",
        "ragged_tma",
    ),
    "mixed_direct": (
        [1] * 5 + [512] * 3,
        [2048, 777, 1500, 33, 4096, 512, 1536, 3512],
        "packed",
        "direct",
    ),
}


@pytest.mark.parametrize("cache_layout", ["combined_hnd", "split_d"])
@pytest.mark.parametrize("sf_offset", [0, 3, 5, 127])
@pytest.mark.parametrize("case", list(_TUNED_CASES))
def test_rubin_paged_gqa_tuned_aot(case, sf_offset, cache_layout):
    """Tuned NVFP4 exports vs fp32, vs the legacy export, and vs offset 0."""
    torch.manual_seed(1999)
    qlens, klens, layout, store = _TUNED_CASES[case]
    total, hq, d = sum(qlens), 64, 128
    q, k, v, kc, vc, table = _paged_inputs(qlens, klens, cache_layout)
    cq, ck = _indptr(qlens), _indptr(klens)
    lengths = torch.tensor(klens, dtype=torch.int32, device="cuda")
    cols = hq * d // 16

    def run(offset, **kwargs):
        o = torch.zeros((total, hq, d // 2), dtype=torch.uint8, device="cuda")
        sf_rows = math.ceil((offset + total) / 128) * 128
        sf = torch.zeros((sf_rows, cols), dtype=q.dtype, device="cuda")

        def call():
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
                output_block_scale=sf,
                o_sf_scale=64.0,
                o_sf_start_index=offset,
                **kwargs,
            )

        _graph_replay(call, o, sf)
        return o, _scale_rows(sf, sf_rows, cols)

    def dequant(o, scales, offset):
        payload = o.reshape(total, -1)
        codes = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(total, -1)
        values = torch.tensor(_E2M1, device="cuda")[codes.long()]
        sf_rows = scales[offset : offset + total].view(torch.float8_e4m3fn).float()
        return values * sf_rows.repeat_interleave(16, dim=1) / 64.0

    tuned = dict(RUBIN_GQA_TUNED_POLICY, query_layout=layout, output_store=store)
    o, scales = run(sf_offset, **tuned)
    # Rows outside [offset, offset + total) of the scale buffer stay untouched.
    assert torch.count_nonzero(scales[:sf_offset]) == 0
    assert torch.count_nonzero(scales[sf_offset + total :]) == 0
    if sf_offset:
        # The offset only moves the scale rows: payload and scale bytes are
        # identical to the offset-0 run.
        o0, scales0 = run(0, **tuned)
        assert torch.equal(o, o0)
        assert torch.equal(scales[sf_offset : sf_offset + total], scales0[:total])
    out = dequant(o, scales, sf_offset)
    ref = _reference(q, k, v, qlens, klens, 0.125 / math.sqrt(d), 2.0, True)
    ref = ref.reshape(total, -1)
    legacy_o, legacy_scales = run(sf_offset)  # shipped export, packed ragged TMA
    legacy = dequant(legacy_o, legacy_scales, sf_offset)
    assert torch.isfinite(out).all()
    err = (out - ref).norm() / ref.norm()
    legacy_err = (legacy - ref).norm() / ref.norm()
    vs_legacy = (out - legacy).norm() / ref.norm()
    print(
        f"RUBIN_GQA_TUNED {case} sf_offset={sf_offset} {cache_layout}: "
        f"rel_l2 {err.item():.4f} (legacy {legacy_err.item():.4f}), "
        f"max_abs {(out - ref).abs().max().item():.4f} "
        f"(legacy {(legacy - ref).abs().max().item():.4f}), "
        f"vs_legacy rel_l2 {vs_legacy.item():.4f} "
        f"max_abs {(out - legacy).abs().max().item():.4f}"
    )
    assert err < 0.2, err.item()
    assert err < legacy_err * 1.05 + 1e-3, (err.item(), legacy_err.item())
    assert vs_legacy < 0.1, vs_legacy.item()
