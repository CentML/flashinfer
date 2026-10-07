"""SM107 MegaMoE capacity ladder helpers (CPU only)."""

import pytest

from flashinfer.moe_ep import (
    sm107_capacity_ladder,
    sm107_max_tokens_per_rank,
    sm107_select_capacity,
)


def test_max_tokens_per_rank_is_ceil():
    assert sm107_max_tokens_per_rank(32768, 2) == 16384
    assert sm107_max_tokens_per_rank(16385, 2) == 8193
    assert sm107_max_tokens_per_rank(16384) == 16384
    with pytest.raises(ValueError):
        sm107_max_tokens_per_rank(0, 1)
    with pytest.raises(ValueError):
        sm107_max_tokens_per_rank(128, 0)


def test_auto_ladder_matches_vllm_without_sp():
    # vLLM sm107_capacity_profiles(16384, "auto")
    assert sm107_capacity_ladder(16384) == [
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]
    assert sm107_capacity_ladder(3000) == [128, 256, 512, 1024, 2048, 3000]
    assert sm107_capacity_ladder(100) == [100]


def test_auto_ladder_capped_by_sp():
    # review_wp7 #1: TEP2 with mnbt 32768 must not allocate a 32768 profile
    ladder = sm107_capacity_ladder(32768, sp_size=2)
    assert ladder[-1] == 16384 and 32768 not in ladder
    assert sm107_capacity_ladder(16385, sp_size=2)[-1] == 8193
    assert sm107_capacity_ladder(32768, sp_size=4)[-1] == 8192


def test_explicit_and_max_specs():
    assert sm107_capacity_ladder(32768, sp_size=2, spec="max") == [16384]
    assert sm107_capacity_ladder(32768, sp_size=2, spec="512, 4096,16384,32768") == [
        512,
        4096,
        16384,
    ]
    assert sm107_capacity_ladder(8192, spec=[1000, 64, 9000]) == [64, 1000, 8192]
    assert sm107_capacity_ladder(8192, spec=None) == sm107_capacity_ladder(8192)
    with pytest.raises(ValueError):
        sm107_capacity_ladder(8192, spec="0,128")


def test_select_capacity():
    ladder = sm107_capacity_ladder(32768, sp_size=2)
    assert sm107_select_capacity(1, ladder) == 128
    assert sm107_select_capacity(128, ladder) == 128
    assert sm107_select_capacity(129, ladder) == 256
    assert sm107_select_capacity(16384, ladder) == 16384
    with pytest.raises(ValueError, match="exceed"):
        sm107_select_capacity(16385, ladder)
    with pytest.raises(ValueError):
        sm107_select_capacity(1, [])
