"""SM107 MegaMoE fast top-k reduce (``shim/fast_reduce.masked_topk_reduce``), any CUDA device.

The masked Triton reduce that replaces the vendored ``TopkReduce`` + the invalid-slot clear must
equal a torch fp32 sum of the valid BF16 combine slots bit for bit (invalid slots, stale NaN bits
included, contribute +0), with and without a fused BF16 addend (== ``y += addend``), with masked
or select loads, and leave the rows past ``num_tokens`` untouched. With router scores the kernel
uses one fma per term, so it is compared with a tolerance against torch's separate rounding.
"""

import pytest
import torch

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA (Triton)"
)


@needs_cuda
@pytest.mark.parametrize("scores", [False, True])
@pytest.mark.parametrize("addend", [False, True])
@pytest.mark.parametrize("masked", ["0", "1"])
def test_masked_topk_reduce_bitwise(monkeypatch, scores, addend, masked):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim.fast_reduce import (
        masked_topk_reduce,
    )

    monkeypatch.setenv("FLASHINFER_SM107_MEGA_FAST_REDUCE_BLOCK", "1,1024,4," + masked)
    torch.manual_seed(0)
    cap, k, h, e, n = 300, 6, 2048, 64, 257
    pre = (torch.randn(cap, k, h, device="cuda") * 3).to(torch.bfloat16)
    ids = torch.randint(0, e, (cap, k), device="cuda", dtype=torch.int32)
    ids[torch.rand(cap, k, device="cuda") < 0.1] = -1
    ids[5] = -1  # a whole padding row
    ids[7, 2] = e + 3  # out-of-range id = invalid route
    bad = ids < 0
    pre.view(torch.int16)[bad] = 0x7FC1  # stale NaN bits in invalid slots
    w = torch.rand(cap, k, device="cuda") if scores else None
    add = torch.randn(cap, h, device="cuda").to(torch.bfloat16) if addend else None
    out = torch.full((cap, h), 7.0, device="cuda", dtype=torch.bfloat16)
    masked_topk_reduce(pre, ids, n, e, out, scores=w, addend=add)
    valid = (ids[:n] >= 0) & (ids[:n] < e)
    terms = torch.where(
        valid[:, :, None], pre[:n].float(), torch.zeros((), device="cuda")
    )
    if scores:
        acc = terms[:, 0] * w[:n, 0, None]
        for j in range(1, k):
            acc = torch.addcmul(acc, terms[:, j], w[:n, j, None])
    else:
        acc = terms[:, 0].clone()
        for j in range(1, k):
            acc = acc + terms[:, j]
    ref = acc.to(torch.bfloat16)
    if addend:
        ref = (ref.float() + add[:n].float()).to(torch.bfloat16)
    got = out[:n]
    if scores:  # the kernel uses fma (one rounding per term); torch rounds the product separately
        torch.testing.assert_close(got.float(), ref.float(), rtol=1e-2, atol=1e-2)
    else:
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    # rows past num_tokens untouched
    assert torch.equal(out[n:], torch.full_like(out[n:], 7.0))
