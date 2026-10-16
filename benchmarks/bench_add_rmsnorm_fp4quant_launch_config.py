"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark: launch configs of the CuTe-DSL fused Add + RMSNorm + FP4 quant kernel

Times explicit (cluster_n, threads_per_row, num_threads) launch configs of
``add_rmsnorm_fp4quant`` against the generic heuristic, and reports which
config ``_compute_launch_config`` selects for each M. Each config's residual
output must equal x + r exactly, and its dequantized FP4 output must match a
float add + RMSNorm reference within the unit tests' tolerance. The configs
are not compared bitwise with each other: cluster_n > 1 configs round h * w
to the input dtype before the block max, so a few block scales and FP4 codes
differ from cluster_n == 1 configs.

The defaults match the NVFP4 call in Qwen3-VL-235B: hidden size 4096, BF16
input, E4M3 scales in the swizzled 128x4 layout, PDL on when supported. The
named configs are the SM107 table in ``AddRMSNormFP4QuantKernel``.

Usage:
    python benchmarks/bench_add_rmsnorm_fp4quant_launch_config.py
    # SM107 table boundaries, hot L2 only, CSV output
    python benchmarks/bench_add_rmsnorm_fp4quant_launch_config.py \
        --m 416 417 1472 1473 5440 5441 --l2 hot --csv out.csv
    # Extra candidate configs
    python benchmarks/bench_add_rmsnorm_fp4quant_launch_config.py \
        --config 1,128,256 --config 2,32,128

Requirements:
    - Blackwell GPU (SM100+), nvidia-cutlass-dsl
    - cupti-python for kernel-only timing (falls back to CUDA events)
"""

import argparse
import csv
import importlib.metadata

import cutlass
import numpy as np
import torch

from flashinfer.cute_dsl.add_rmsnorm_fp4quant import (
    AddRMSNormFP4QuantKernel,
    _get_compiled_kernel,
    get_sm_version,
)
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import device_support_pdl, get_device_sm_count

BLOCK_SIZE = 16  # NVFP4
SCALE_FORMAT = "e4m3"
EPS = 1e-6
DEFAULT_M = [256, 1024, 4096, 8192, 16384, 21504]
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
# E2M1 code -> value; codes 8-15 are the negatives of codes 0-7.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES += tuple(-v for v in E2M1_VALUES)
# Same tiered tolerance as tests/norm/test_add_rmsnorm_fp4_quant_cute_dsl.py:
# TIGHT_FRACTION of the elements within the tight bound, all within the loose.
TIGHT_RTOL, TIGHT_ATOL, TIGHT_FRACTION = 0.3, 0.5, 0.99
LOOSE_RTOL, LOOSE_ATOL = 0.5, 2.0


def generic_launch_config(hidden_size, cutlass_dtype, sm_version):
    """Launch config of the generic heuristic (SM107 before the table)."""
    kernel = AddRMSNormFP4QuantKernel
    cluster_n = kernel._compute_cluster_n(hidden_size, cutlass_dtype, sm_version)
    h_per_cta = hidden_size // cluster_n
    return (
        cluster_n,
        kernel._compute_threads_per_row(h_per_cta),
        kernel._compute_num_threads(h_per_cta),
    )


def parse_config(text):
    values = tuple(int(v) for v in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            f"expected cluster_n,threads_per_row,num_threads, got {text!r}"
        )
    return values


def reference_norm(x, r, w):
    """Float add + RMSNorm of the BF16/FP16 residual sum, as the kernel computes."""
    h = (x + r).float()
    rstd = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return h * rstd * w.float()


def make_inputs(m, hidden_size, dtype, swizzled):
    """Allocate kernel inputs/outputs in the layout add_rmsnorm_fp4quant uses."""
    num_sf = hidden_size // BLOCK_SIZE
    if swizzled:
        num_m_tiles = (m + 127) // 128
        num_k_tiles = (num_sf + 3) // 4
        sf_shape = (num_m_tiles * num_k_tiles * 512,)
    else:
        sf_shape = (m, num_sf)
    x = torch.randn(m, hidden_size, device="cuda", dtype=dtype)
    r = torch.randn(m, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    # Per-tensor NVFP4 scale from the output amax, as a calibrated model uses,
    # so block scales do not saturate at the E4M3 max.
    amax = reference_norm(x, r, weight).abs().max().item()
    return {
        "x": x,
        "r": r,
        "w": weight,
        "y": torch.zeros(m, hidden_size // 2, device="cuda", dtype=torch.uint8),
        "s": torch.zeros(sf_shape, device="cuda", dtype=torch.uint8),
        "s_unswizzled": torch.empty(m, num_sf, device="cuda", dtype=torch.uint8),
        # Not written unless output_norm=True; the API passes the weight too.
        "y_norm": weight,
        "global_scale": torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax],
            device="cuda",
            dtype=torch.float32,
        ),
    }


def dequantize(t, m, hidden_size, swizzled):
    """Dequantize the packed E2M1 output with its E4M3 block scales."""
    num_sf = hidden_size // BLOCK_SIZE
    s = t["s"]
    if swizzled:
        # 128x4 tiles stored as [m_tile][k_tile][32][4][4]; the row within a
        # tile is inner_m * 32 + outer_m (see the kernel's swizzled_offset).
        num_m_tiles = (m + 127) // 128
        num_k_tiles = (num_sf + 3) // 4
        s = s.view(num_m_tiles, num_k_tiles, 32, 4, 4).transpose(1, 3)
        s = s.reshape(num_m_tiles * 128, num_k_tiles * 4)[:m, :num_sf]
    scales = s.contiguous().view(torch.float8_e4m3fn).float()
    y = t["y"]
    # The first element of each pair is in the low nibble.
    codes = torch.stack((y & 0xF, y >> 4), dim=-1).view(m, num_sf, BLOCK_SIZE)
    lut = torch.tensor(E2M1_VALUES, device=y.device)
    values = lut[codes.long()] * scales.unsqueeze(-1)
    return values.view(m, hidden_size) / t["global_scale"]


def run_kernel(fn, t, m):
    fn(
        t["x"],
        t["r"],
        t["w"],
        t["y"],
        t["s"],
        t["s_unswizzled"],
        t["y_norm"],
        t["global_scale"],
        m,
        EPS,
    )


def check_output(fn, base, m, hidden_size, swizzled, expected_r, ref):
    """Run fn on a copy of base; check the residual and the dequantized output."""
    t = dict(base)
    t["r"] = base["r"].clone()
    t["y"] = torch.zeros_like(base["y"])
    t["s"] = torch.zeros_like(base["s"])
    run_kernel(fn, t, m)
    torch.cuda.synchronize()
    if not torch.equal(t["r"], expected_r):
        return "residual-mismatch"
    diff = (dequantize(t, m, hidden_size, swizzled) - ref).abs()
    ref_abs = ref.abs()
    tight = ((diff <= TIGHT_ATOL) | (diff <= TIGHT_RTOL * ref_abs)).float().mean()
    loose = ((diff <= LOOSE_ATOL) | (diff <= LOOSE_RTOL * ref_abs)).all()
    if tight.item() < TIGHT_FRACTION or not loose.item():
        return f"tolerance(tight={tight.item():.4f},max_diff={diff.max().item():.3f})"
    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark add_rmsnorm_fp4quant launch configs."
    )
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--config",
        type=parse_config,
        action="append",
        default=[],
        help="extra cluster_n,threads_per_row,num_threads config (repeatable)",
    )
    parser.add_argument(
        "--sm-version",
        type=int,
        default=None,
        help="SM version for the 'selected' column (default: current device)",
    )
    parser.add_argument("--l2", choices=["cold", "hot", "both"], default="both")
    parser.add_argument("--linear-sf", action="store_true", help="row-major scales")
    parser.add_argument("--no-pdl", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv", default=None, help="write results to this file")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    cutlass_dtype = cutlass.BFloat16 if args.dtype == "bf16" else cutlass.Float16
    is_fp16 = dtype == torch.float16
    swizzled = not args.linear_sf
    enable_pdl = (not args.no_pdl) and device_support_pdl(device)
    device_sm = get_sm_version(device)
    sm_version = args.sm_version if args.sm_version is not None else device_sm
    hidden_size = args.hidden_size

    try:
        dsl_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        dsl_version = "unknown"
    print(
        f"GPU: {torch.cuda.get_device_name(device)} (SM{device_sm}, "
        f"{get_device_sm_count(device)} SMs), CuTe DSL {dsl_version}"
    )
    print(
        f"H={hidden_size} dtype={args.dtype} block_size={BLOCK_SIZE} "
        f"sf={'swizzled' if swizzled else 'linear'} pdl={enable_pdl} "
        f"cuda_graph={args.cuda_graph} selection_sm={sm_version}"
    )

    configs = {"generic": generic_launch_config(hidden_size, cutlass_dtype, device_sm)}
    configs.update(AddRMSNormFP4QuantKernel._SM107_H4096_LAUNCH_CONFIGS)
    selected = {
        m: AddRMSNormFP4QuantKernel._compute_launch_config(
            hidden_size, m, cutlass_dtype, sm_version, BLOCK_SIZE, SCALE_FORMAT
        )
        for m in args.m
    }
    for cfg in [*args.config, *selected.values()]:
        if cfg not in configs.values():
            configs[f"c{cfg[0]}_{cfg[1]}_{cfg[2]}"] = cfg
    names = list(configs)
    print("Configs (cluster_n, threads_per_row, num_threads):")
    for name in names:
        print(f"  {name:<10} {configs[name]}")

    kernels = {}
    for name in names:
        try:
            kernels[name] = _get_compiled_kernel(
                hidden_size,
                BLOCK_SIZE,
                is_fp16,
                device_sm,
                SCALE_FORMAT,
                swizzled,
                configs[name],
                False,  # output_both_sf_layouts
                enable_pdl,
                False,  # output_norm
            )
        except Exception as e:  # invalid geometry for this H / device
            print(f"  {name}: compile failed: {e}")
    if not kernels:
        raise SystemExit("No launch config compiled.")

    l2_modes = ["cold", "hot"] if args.l2 == "both" else [args.l2]
    rows = []
    for m in args.m:
        base = make_inputs(m, hidden_size, dtype, swizzled)
        # Prefer a table name over "generic" when both have the same geometry.
        selected_name = next(n for n in reversed(names) if configs[n] == selected[m])
        expected_r = base["x"] + base["r"]
        ref = reference_norm(base["x"], base["r"], base["w"])
        status = {
            name: check_output(fn, base, m, hidden_size, swizzled, expected_r, ref)
            for name, fn in kernels.items()
        }
        del base, expected_r, ref
        for l2 in l2_modes:
            times = {}
            for name, fn in kernels.items():
                t = make_inputs(m, hidden_size, dtype, swizzled)
                times[name] = (
                    np.median(
                        bench_gpu_time(
                            lambda fn=fn, t=t, m=m: run_kernel(fn, t, m),
                            dry_run_iters=args.warmup,
                            repeat_iters=args.iters,
                            enable_cupti=True,
                            use_cuda_graph=args.cuda_graph,
                            cold_l2_cache=(l2 == "cold"),
                        )
                    )
                    * 1e3
                )
            rows.append(
                {
                    "m": m,
                    "l2": l2,
                    "selected": selected_name,
                    "times_us": times,
                    "status": status,
                }
            )

    print()
    print(
        "gen/sel = generic / selected time (>1: selection beats generic); "
        "best/sel = best / selected time (1: selection is the best config)"
    )
    for l2 in l2_modes:
        print()
        print(f"Median kernel time (us), {l2} L2")
        header = f"{'M':>6} " + " ".join(f"{n:>10}" for n in names)
        header += f" {'best':>8} {'selected':>9} {'gen/sel':>8} {'best/sel':>8}"
        print(header)
        print("-" * len(header))
        for row in rows:
            if row["l2"] != l2:
                continue
            times = row["times_us"]
            line = f"{row['m']:>6} " + " ".join(
                f"{times[n]:>10.2f}" if n in times else f"{'n/a':>10}" for n in names
            )
            best = min(times, key=times.get)
            sel = row["selected"]
            gen_sel = best_sel = "n/a"
            if sel in times:
                best_sel = f"{times[best] / times[sel]:.3f}"
                if "generic" in times:
                    gen_sel = f"{times['generic'] / times[sel]:.3f}"
            line += f" {best:>8} {sel:>9} {gen_sel:>8} {best_sel:>8}"
            print(line)

    bad = {
        (row["m"], name): s
        for row in rows
        for name, s in row["status"].items()
        if s != "ok"
    }
    print()
    check = (
        f"Output check vs float reference ({TIGHT_FRACTION:.0%} within "
        f"rtol={TIGHT_RTOL}/atol={TIGHT_ATOL}, all within "
        f"rtol={LOOSE_RTOL}/atol={LOOSE_ATOL})"
    )
    if bad:
        print(f"{check} FAILED:")
        for (m, name), s in sorted(bad.items()):
            print(f"  M={m} {name}: {s}")
    else:
        print(f"{check}: all ok")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["m", "l2", "config", "launch_config", "median_us", "selected", "check"]
            )
            for row in rows:
                for name, t in row["times_us"].items():
                    writer.writerow(
                        [
                            row["m"],
                            row["l2"],
                            name,
                            "x".join(map(str, configs[name])),
                            f"{t:.3f}",
                            int(name == row["selected"]),
                            row["status"][name],
                        ]
                    )
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
