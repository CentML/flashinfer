"""SM107 MegaMoE route staging with the caller's padding mask (``MoEEpTensors.route_padding_mask``).

Rows flagged as padding stage no routes (every top-k id -1); ``keep_first_route`` keeps row 0's
routes so no rank stages a zero-route plane. The one-launch Triton staging (CUDA) and the torch
fallback (CPU tensors) must both equal masking the ids before staging.
"""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import stage_routing_plane


def _ws(capacity, topk, experts=16, world=1, device="cpu"):
    return SimpleNamespace(
        config=SimpleNamespace(num_total_experts=experts, world_size=world),
        topk_idx=torch.full((capacity, topk), -1, dtype=torch.int32, device=device),
        topk_weights=torch.zeros((capacity, topk), dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("keep_first", [False, True])
def test_padding_mask_cpu_fallback(keep_first):
    n, k = 9, 4
    ids = torch.stack([torch.randperm(16)[:k] for _ in range(n)]).to(torch.int32)
    w = torch.rand(n, k)
    pad = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=torch.bool)
    routed, out_ids, out_w = stage_routing_plane(
        ids, w, _ws(16, k), pad_mask=pad, keep_first_row=keep_first
    )
    assert routed is False  # CPU tensors: torch fallback
    ref = torch.where(pad.unsqueeze(1), torch.full_like(ids, -1), ids)
    if keep_first:
        ref[0] = ids[0]
    assert torch.equal(out_ids, ref)
    assert torch.equal(out_w, w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
def test_route_staging_padding_mask_triton():
    from flashinfer.moe_ep.backends.mega.kernel.sm107.route_stage import stage_routes

    torch.manual_seed(1)
    n, k, e, cap = 150, 6, 384, 256
    ids = torch.stack([torch.randperm(e)[:k] for _ in range(n)]).to(torch.int32).cuda()
    w = torch.rand(n, k, device="cuda")
    pad = torch.rand(n, device="cuda") < 0.3
    pad[0] = True
    ref_ids = torch.where(pad.unsqueeze(1), torch.full_like(ids, -1), ids)
    ref_ids[:1] = ids[:1]
    oa = torch.full((cap, k), 9, dtype=torch.int32, device="cuda")
    ob = oa.clone()
    wa = torch.zeros(cap, k, device="cuda")
    wb = wa.clone()
    assert stage_routes(ref_ids.contiguous(), w, oa, wa, e)
    assert stage_routes(ids, w, ob, wb, e, pad_mask=pad, keep_first_row=True)
    assert torch.equal(oa, ob)
    rows = (oa[:n] >= 0).any(1)
    assert torch.equal(wa[:n][rows], wb[:n][rows])
