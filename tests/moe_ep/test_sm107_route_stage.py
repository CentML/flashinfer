"""One-launch SM107 MegaMoE routing staging (``route_stage.stage_routes``).

Regression tests for review_wp6 I3: the staging grid covers
``cdiv(capacity, 64) * 64`` rows, and the weight store used to be masked by
``row < num_tokens`` only, so ``num_tokens > capacity`` with ``capacity % 64 != 0``
wrote past ``topk_weights_out``. The host wrapper now rejects
``num_tokens > capacity`` and the kernel masks both stores with the capacity.
Guard rows after the capacity detect any out-of-bounds write.
"""

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm107 import route_stage
from flashinfer.moe_ep.backends.mega.kernel.sm107.route_stage import stage_routes

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="stage_routes needs a CUDA device"
)

NUM_EXPERTS = 64
TOP_K = 6
GUARD = 192  # > one 64-row block past any capacity used below
ID_SENTINEL = -7
W_SENTINEL = 123.0


def _routes(num_tokens, ids_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=g)
    ids = scores.topk(TOP_K, dim=-1).indices  # unique per row
    if num_tokens:
        ids[::5, -1] = -1  # some dropped routes
    w = torch.rand(num_tokens, TOP_K, device="cuda", generator=g, dtype=torch.float32)
    return ids.to(ids_dtype).contiguous(), w.to(torch.bfloat16).contiguous()


def _outputs(capacity):
    ids_full = torch.full(
        (capacity + GUARD, TOP_K), ID_SENTINEL, dtype=torch.int32, device="cuda"
    )
    w_full = torch.full(
        (capacity + GUARD, TOP_K), W_SENTINEL, dtype=torch.float32, device="cuda"
    )
    return ids_full, w_full


def _assert_guard_intact(ids_full, w_full, capacity):
    assert (ids_full[capacity:] == ID_SENTINEL).all(), "id store past capacity"
    assert (w_full[capacity:] == W_SENTINEL).all(), "weight store past capacity"


@requires_cuda
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "capacity,num_tokens",
    [(100, 37), (100, 100), (1, 1), (65, 64), (128, 0), (130, 129), (16, 16)],
)
def test_stage_routes_within_capacity(capacity, num_tokens, ids_dtype):
    ids, w = _routes(num_tokens, ids_dtype, seed=capacity + num_tokens)
    ids_full, w_full = _outputs(capacity)
    ids_out, w_out = ids_full[:capacity], w_full[:capacity]
    assert stage_routes(ids, w, ids_out, w_out, NUM_EXPERTS) is True
    torch.cuda.synchronize()
    assert torch.equal(ids_out[:num_tokens], ids.to(torch.int32))
    assert torch.equal(w_out[:num_tokens], w.float())
    assert (ids_out[num_tokens:] == -1).all()
    _assert_guard_intact(ids_full, w_full, capacity)


@requires_cuda
@pytest.mark.parametrize(
    "capacity,num_tokens", [(100, 101), (100, 130), (1, 2), (65, 200), (16, 37)]
)
def test_stage_routes_rejects_tokens_beyond_capacity(capacity, num_tokens):
    ids, w = _routes(num_tokens, torch.int32, seed=num_tokens)
    ids_full, w_full = _outputs(capacity)
    with pytest.raises(ValueError, match="exceed the workspace capacity"):
        stage_routes(ids, w, ids_full[:capacity], w_full[:capacity], NUM_EXPERTS)
    torch.cuda.synchronize()
    # Nothing was launched: the capacity rows are untouched as well.
    assert (ids_full == ID_SENTINEL).all() and (w_full == W_SENTINEL).all()


@requires_cuda
@pytest.mark.parametrize("capacity,num_tokens", [(100, 130), (1, 63), (65, 128)])
def test_stage_routes_kernel_masks_capacity(capacity, num_tokens):
    """The kernel itself never stores past ``capacity`` (defense in depth for
    callers that launch it directly)."""
    import triton

    ids, w = _routes(num_tokens, torch.int32, seed=3)
    ids_full, w_full = _outputs(capacity)
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    grid = (triton.cdiv(capacity, 64),)
    route_stage._get_kernel()[grid](
        ids,
        w,
        ids_full,
        w_full,
        flag,
        flag,  # pad_ptr (unused: HAS_PAD=False)
        num_tokens,
        capacity,
        NUM_EXPERTS,
        TOPK=TOP_K,
        KP=triton.next_power_of_2(TOP_K),
        BLOCK_R=64,
        CHECK=False,
        HAS_PAD=False,
        KEEP_FIRST=False,
    )
    torch.cuda.synchronize()
    assert torch.equal(ids_full[:capacity], ids[:capacity])
    assert torch.equal(w_full[:capacity], w[:capacity].float())
    _assert_guard_intact(ids_full, w_full, capacity)


def test_stage_routes_falls_back_on_mismatched_weight_buffer():
    """A weight staging buffer that does not match the id buffer is not staged
    by the one-launch path (the caller falls back to the torch path)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    ids, w = _routes(8, torch.int32)
    ids_out = torch.empty(16, TOP_K, dtype=torch.int32, device="cuda")
    w_out = torch.empty(15, TOP_K, dtype=torch.float32, device="cuda")
    assert stage_routes(ids, w, ids_out, w_out, NUM_EXPERTS) is False
