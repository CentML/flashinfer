"""EP stress of the SM107 MegaMoE phase-interleave FC1/FC2 scheduler (deadlock regression).

The vendored ``PhaseInterleavedFc12Scheduler`` used to set ``fc1_exhausted`` from a CTA-local
peek of the global FC1 work-ID counter. When the runtime FC1 stream is shorter than the static
FC1 watermark (tiny per-rank work: DP dummy batches, a single decode token on one rank) and
the clusters of a rank start at different times (a side-stream kernel holding SMs), two CTAs
of one cluster could observe different counter values, disagree on ``fc1_exhausted`` and
deadlock the whole EP group: one rank two CTAs short of its tail grid sync, its peers waiting
in the NVLink drain barrier. ``VENDOR.md`` records the FlashInfer fix.

Each case below is a serving pattern that hung within 1 - 2,670 graph replays before the fix.
The layer mirrors vLLM's ``VLLM_FI_MEGA_MOE_SHARED=overlap`` composition: NVFP4 DeepSeek-V4.1
routed experts (384 experts, top-6, intermediate 2304, hidden 4096) with the built-in
heuristic knobs (``default_knobs``: phase-interleave schedule), the megakernel SM budget
vLLM uses next to the side stream (180 SMs up to 2048 tokens/rank, 148 above), and a BF16
MLP on a side stream forked after the input staging and enqueued after the megakernel:

* ``boot``: capacity 3072, every rank keeps the routes of one row only (identical on all
  ranks), the rest is padding: vLLM's dummy batches during start-up profiling;
* ``c1_skew``: capacity 128, 6 tokens, only the last rank routes, plus a busy kernel that
  holds 96 CTAs for 30 us when the megakernel launches;
* ``random_live``: capacity 1024, every replay each rank keeps a random number of routed
  rows (zero, tiny, small or any).

A graph of ``LAYERS`` MoE calls is replayed ``FLASHINFER_SM107_PI_STRESS_REPLAYS`` times
(default 500) with fresh routes. A deadlocked persistent kernel never returns, so a per-rank
watchdog ends the process (exit code 3) when no replay finishes within
``FLASHINFER_SM107_PI_STRESS_STALL_S`` seconds (default 120); torchrun then reports the
failure. Runs ONLY under its own torchrun invocation (one GPU per rank, one NVLink domain)::

    NVSHMEM_SYMMETRIC_SIZE=8G CUTE_DSL_ARCH=sm_107a torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_sm107_phase_interleave_stress_multirank.py -v \\
        -m "gpu_2 and arch_rubin"
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time

import pytest
import torch

pytest.importorskip("flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe")

from flashinfer.moe_ep import (  # noqa: E402
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
    Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel  # noqa: E402
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (  # noqa: E402
    default_knobs,
)

NUM_EXPERTS = 384
HIDDEN = 4096
INTERMEDIATE = 2304
TOP_K = 6
SHARED_INTERMEDIATE = 2048
# vLLM's megakernel SM budget for an overlapped shared expert (tokens/rank, SMs)
MAX_SM_COUNT = ((2048, 180), (1 << 20, 148))
LAYERS = 40

# name -> (capacity, tokens, routing pattern, busy kernel (CTAs, us) or None)
CASES = {
    "boot": (3072, 2280, "first_row", None),
    "c1_skew": (128, 6, "last_rank", (96, 30)),
    "random_live": (1024, 1024, "random_live", None),
}


def _launcher_ranks() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _spin_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _spin(out_ptr, ns):
        pid = tl.program_id(0)
        t0 = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l,r",
            [pid],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
        t = t0
        while t - t0 < ns:
            t = tl.inline_asm_elementwise(
                "mov.u64 $0, %globaltimer;",
                "=l,r",
                [pid],
                dtype=tl.int64,
                is_pure=False,
                pack=1,
            )
        tl.store(out_ptr + pid, t - t0)

    return _spin


def _random_nvfp4_pack(local_experts: int, gen: torch.Generator) -> MoEWeightPack:
    """Random NVFP4 bytes and E4M3 block scales: the test checks liveness, not numerics."""

    def data(*shape):
        return torch.randint(
            0, 256, shape, device="cuda", generator=gen, dtype=torch.int32
        ).to(torch.uint8)

    def scales(*shape):
        return (torch.rand(shape, device="cuda", generator=gen) * 1.5 + 0.5).to(
            torch.float8_e4m3fn
        )

    e, h, i = local_experts, HIDDEN, INTERMEDIATE
    return MoEWeightPack(
        w13=data(e, 2 * i, h // 2),
        w2=data(e, h, i // 2),
        w13_scale=scales(e, 2 * i, h // 16),
        w2_scale=scales(e, h, i // 16),
    )


class _Watchdog:
    """Ends the process when no replay completes within ``stall_s`` seconds."""

    def __init__(self, tag: str, stall_s: float, pytestconfig):
        self.tag, self.stall_s, self.config = tag, stall_s, pytestconfig
        self.replays, self.last = 0, time.monotonic()
        self.done = False
        threading.Thread(target=self._run, daemon=True).start()

    def tick(self):
        self.replays += 1
        self.last = time.monotonic()

    def _run(self):
        while not self.done:
            time.sleep(1.0)
            if not self.done and time.monotonic() - self.last > self.stall_s:
                rank, _ = _launcher_ranks()
                # os._exit skips pytest's report: release the output capture first
                capman = self.config.pluginmanager.getplugin("capturemanager")
                if capman is not None:
                    capman.suspend_global_capture(in_=False)
                print(
                    f"SM107 phase-interleave stress {self.tag}: rank {rank} STALLED after "
                    f"{self.replays} graph replays ({self.replays * LAYERS} MoE launches); "
                    "the megakernel did not return (EP deadlock)",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(3)


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("case", list(CASES))
def test_phase_interleave_overlap_does_not_deadlock(case, pytestconfig):
    import torch.distributed as dist

    rank, world = _launcher_ranks()
    if world < 2:
        pytest.skip("requires torchrun with >= 2 ranks")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires SM107 GPUs")
    capacity, tokens, pattern, spin = CASES[case]
    replays = int(os.environ.get("FLASHINFER_SM107_PI_STRESS_REPLAYS", "500"))
    stall_s = float(os.environ.get("FLASHINFER_SM107_PI_STRESS_STALL_S", "120"))

    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    local_experts = NUM_EXPERTS // world
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    # The built-in heuristic knob profile (what knobs="cache" falls back to): phase-interleave
    # schedule. The SM budget makes the launch uniform and leaves SMs to the side-stream MLP.
    knobs = default_knobs(capacity, quant_kind="nvfp4")
    assert knobs["schedule_policy"][0] == "phase_interleave"
    cfg = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=TOP_K,
        gate_up_clamp=10.0,
        knobs=knobs,
        input_norm_const=1.0,
        max_sm_count=MAX_SM_COUNT,
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(cfg).runtime_requirements(bootstrap)
    )
    layer = None
    try:
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                rank=rank, world_size=world, auto_bootstrap=False
            ),
            fleet_params=FleetParams(
                num_experts=NUM_EXPERTS,
                max_tokens_per_rank=capacity,
                token_hidden_size=HIDDEN,
            ),
            weights=_random_nvfp4_pack(local_experts, gen),
            backend=MegaConfig(megakernel=cfg),
        )
        x = torch.randn(tokens, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
        ids = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device="cuda")
        weights = torch.full((tokens, TOP_K), 1.0 / TOP_K, device="cuda")
        t = MoEEpTensors(x, ids, weights)

        # Routes: identical across ranks for the boot pattern, per rank otherwise.
        route_gen = torch.Generator(device="cuda").manual_seed(
            777 if pattern == "first_row" else 777 + rank
        )
        live_rng = random.Random(4242 + 31 * rank)

        def draw(rows: int):
            return (
                torch.rand(rows, NUM_EXPERTS, device="cuda", generator=route_gen)
                .topk(TOP_K, dim=1)
                .indices.to(torch.int32)
            )

        def reroute():
            ids.fill_(-1)
            if pattern == "first_row":
                live = 1
            elif pattern == "last_rank":
                live = tokens if rank == world - 1 else 0
            else:
                u = live_rng.random()
                if u < 0.25:
                    live = 0
                elif u < 0.55:
                    live = live_rng.randint(1, min(8, tokens))
                elif u < 0.80:
                    live = live_rng.randint(1, max(1, tokens // 8))
                else:
                    live = live_rng.randint(1, tokens)
            if live:
                ids[:live].copy_(draw(live))

        w1 = (
            torch.randn(
                HIDDEN, 2 * SHARED_INTERMEDIATE, device="cuda", generator=gen
            ).to(torch.bfloat16)
            * HIDDEN**-0.5
        )
        w2 = (
            torch.randn(SHARED_INTERMEDIATE, HIDDEN, device="cuda", generator=gen).to(
                torch.bfloat16
            )
            * SHARED_INTERMEDIATE**-0.5
        )
        side = torch.cuda.Stream()
        events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(LAYERS)]
        spin_kernel = _spin_kernel() if spin is not None else None
        spin_stream = torch.cuda.Stream()
        spin_events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(LAYERS)]
        spin_out = torch.zeros(spin[0] if spin else 1, dtype=torch.int64, device="cuda")

        # vLLM's overlap order (sm107_compute_with_shared): stage the inputs, record the
        # fork, enqueue the megakernel, then the shared MLP on the side stream (it runs on
        # the SMs the megakernel leaves free), join, add.
        kernel = layer._kernel
        transformed = layer._get_transformed_weights()
        workspace = layer._ensure_workspace()
        assert workspace.config.schedule_policy[0] == "phase_interleave"

        def step():
            out = None
            for i, (fork, join) in enumerate(events):
                if spin_kernel is not None:
                    # busy kernel holding SMs when the megakernel's clusters launch
                    spin_fork, spin_join = spin_events[i]
                    spin_fork.record()
                    with torch.cuda.stream(spin_stream):
                        spin_fork.wait()
                        spin_kernel[(spin[0],)](spin_out, spin[1] * 1000, num_warps=4)
                        spin_join.record()
                kernel.stage_inputs(t, workspace, quantize_input=True)
                fork.record()
                y = kernel.compute(workspace, transformed, output=None)
                with torch.cuda.stream(side):
                    fork.wait()
                    u = x @ w1
                    shared = (
                        torch.nn.functional.silu(u[:, :SHARED_INTERMEDIATE])
                        * u[:, SHARED_INTERMEDIATE:]
                    ) @ w2
                    join.record()
                join.wait()
                if spin_kernel is not None:
                    spin_join.wait()
                out = y + shared
            return out

        reroute()
        step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        torch.cuda.synchronize()
        dist.barrier()

        watchdog = _Watchdog(case, stall_s, pytestconfig)
        try:
            for _ in range(replays):
                reroute()
                graph.replay()
                torch.cuda.synchronize()
                watchdog.tick()
        finally:
            watchdog.done = True
        dist.barrier()
        assert watchdog.replays == replays
        del graph
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
