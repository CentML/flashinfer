"""Workspace capacity ladder for the SM107 MegaMoE backends (host helpers).

A MegaMoE workspace is sized for a token *capacity* per rank; serving frameworks
keep a ladder of workspaces (one per capacity profile) and pick, per step, the
smallest profile that holds the step's per-rank token count (vLLM
``flashinfer_moe_ep_mega_cutedsl``: ``VLLM_FI_MEGA_MOE_CAPACITIES``).

With sequence parallelism (SP, e.g. TP2 deployed as TEP2) every rank sees at most
``ceil(max_num_batched_tokens / sp_size)`` tokens, so profiles above that are
unreachable and only cost memory (review_wp7 #1: a 32768-token profile at TEP2
with max_num_batched_tokens 32768, where every step selects <= 16384).
:func:`sm107_capacity_ladder` caps the ladder there; :func:`sm107_select_capacity`
returns the profile for a step (always >= the token count, see review_wp6 I3).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

SM107_MIN_CAPACITY = 128


def sm107_max_tokens_per_rank(max_num_batched_tokens: int, sp_size: int = 1) -> int:
    """Largest per-rank token count: ``ceil(max_num_batched_tokens / sp_size)``."""
    mnbt, sp = int(max_num_batched_tokens), int(sp_size)
    if mnbt <= 0:
        raise ValueError(f"max_num_batched_tokens must be positive, got {mnbt}")
    if sp <= 0:
        raise ValueError(f"sp_size must be positive, got {sp}")
    return -(-mnbt // sp)


def sm107_capacity_ladder(
    max_num_batched_tokens: int,
    *,
    sp_size: int = 1,
    spec: Union[str, Iterable[int], None] = "auto",
    min_capacity: int = SM107_MIN_CAPACITY,
) -> List[int]:
    """Sorted capacity profiles (tokens per rank) capped at the per-rank maximum.

    ``spec`` follows vLLM's ``VLLM_FI_MEGA_MOE_CAPACITIES``:

    * ``"auto"`` (default): powers of two from ``min_capacity`` below the cap,
      plus the cap itself;
    * ``"max"``: only the cap;
    * a comma-separated string or an iterable of ints: those profiles; values at
      or above the cap are unreachable and dropped, and the cap is always added
      so every legal step has a profile.

    The cap is :func:`sm107_max_tokens_per_rank` ``(max_num_batched_tokens,
    sp_size)``; with ``sp_size=1`` the ladder equals vLLM's
    ``sm107_capacity_profiles(max_num_batched_tokens, spec)``.
    """
    cap = sm107_max_tokens_per_rank(max_num_batched_tokens, sp_size)
    caps = {cap}
    if spec is None:
        spec = "auto"
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "auto"):
            c = int(min_capacity)
            if c <= 0:
                raise ValueError(f"min_capacity must be positive, got {c}")
            while c < cap:
                caps.add(c)
                c *= 2
            return sorted(caps)
        if s == "max":
            return sorted(caps)
        values: Sequence[int] = [int(v) for v in s.split(",") if v.strip()]
    else:
        values = [int(v) for v in spec]
    for v in values:
        if v <= 0:
            raise ValueError(f"capacity profiles must be positive, got {v}")
        if v < cap:
            caps.add(v)
    return sorted(caps)


def sm107_select_capacity(num_tokens: int, ladder: Sequence[int]) -> int:
    """Smallest profile in ``ladder`` that holds ``num_tokens`` tokens per rank.

    Raises ``ValueError`` when ``num_tokens`` exceeds the largest profile (the
    staging path rejects ``num_tokens > capacity``; never truncate).
    """
    n = int(num_tokens)
    if not ladder:
        raise ValueError("empty capacity ladder")
    for c in sorted(ladder):
        if n <= c:
            return int(c)
    raise ValueError(
        f"{n} tokens per rank exceed the largest capacity profile {max(ladder)}"
    )


__all__ = [
    "SM107_MIN_CAPACITY",
    "sm107_capacity_ladder",
    "sm107_max_tokens_per_rank",
    "sm107_select_capacity",
]
