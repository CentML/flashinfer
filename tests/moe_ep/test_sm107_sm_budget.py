"""SM107 MegaMoE SM budget (``max_sm_count``) resolved per workspace capacity (CPU).

``max_sm_count`` is an int (every capacity) or ``((max_tokens, sm_count), ...)`` pairs, so a
serving framework can leave SMs to a concurrent side-stream kernel (e.g. vLLM's overlapped
shared-expert MLP, ``VLLM_FI_MEGA_MOE_SHARED=overlap``) only where the megakernel does not
need them.
"""

from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import (
    max_sm_count_key,
    resolve_max_sm_count,
)


def test_resolve_max_sm_count():
    assert resolve_max_sm_count(None, 128) is None
    assert resolve_max_sm_count(0, 128) is None
    assert resolve_max_sm_count(180, 16384) == 180
    spec = [(16384, 148), (2048, 180)]
    assert resolve_max_sm_count(spec, 128) == 180
    assert resolve_max_sm_count(spec, 2048) == 180
    assert resolve_max_sm_count(spec, 4096) == 148
    assert resolve_max_sm_count(spec, 32768) == 148  # beyond the last pair
    assert resolve_max_sm_count([(1024, 0), (65536, 160)], 512) is None
    assert max_sm_count_key(spec) == ((2048, 180), (16384, 148))
    hash(max_sm_count_key(spec))
