# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Explicit AOT-only entry points for Rubin ViT and paged GQA.

These specializations are not implemented by the public FMHA JIT source. A
missing artifact is an error, not permission to silently compile a different
kernel. Warm up before CUDA graph capture (or preinstall the cubin wheel).

Exports load from the cute-dsl FMHA collection pinned by
``ArtifactPath.DSL_FMHA`` in ``flashinfer/artifacts.py``. The MR199 collection
(a10580e7) supplies only D72/D80 ViT and BF16 NHD paged GQA. HND, fused NVFP4
output, the multi-sequence ragged page-table fix, the tuned paged GQA and the
persistent ViT variants need a later cubin_publishing collection.
FLASHINFER_DSL_FMHA_LOCAL_DIR, when set, replaces the pinned collection with a
directory of exports, for example a build that is not published yet.
"""

import math
import os
from collections.abc import Sequence
from types import MappingProxyType

import torch
from cutlass.cute.typing import Float32, Int32

from .fmha import DSL_FMHA_ARTIFACT_PATH, _get_gpu_arch, get_cute_dsl_fmha_kernel


def _require_tensor(tensor, name, device, dtype, ndim, *, compact=True, align=16):
    if tensor.device != device or tensor.dtype != dtype or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D {dtype} tensor on {device}")
    if compact and not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; implicit copies are not supported"
        )
    if tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")


def _require_rubin(q):
    if q.device.type != "cuda" or _get_gpu_arch(q.device) != "sm_107a":
        raise ValueError("These AOT FMHA specializations require a Rubin SM107 GPU")


def _positive_scalar(value, name):
    # Do not use float(tensor): that would synchronize or fail during capture.
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite host scalar")
    return float(value)


def _scales(scale_q, scale_k, scale_v, scale_o, sm_scale, head_dim):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    qk = (
        _positive_scalar(scale_q, "scale_q")
        * _positive_scalar(scale_k, "scale_k")
        * _positive_scalar(sm_scale, "sm_scale")
    )
    output = _positive_scalar(scale_v, "scale_v") / _positive_scalar(scale_o, "scale_o")
    return Float32(qk * math.log2(math.e)), Float32(qk), Float32(output)


_GQA_Q_ROWS_PER_CTA = 256  # mma_tiler M (128) x 2 O tiles per CTA

#: Paged-GQA kernel options measured fastest for Qwen3-VL on VR200 with the
#: current DKG exports: cluster launch with CLC Work IDs for TMA stores
#: (static scheduling for direct stores), no softmax sequence barrier, the
#: default skip-correction threshold, the MMA row sum and the packed NVFP4
#: epilogue (ignored for BF16 output). An older reference build used
#: threshold 0, which is slower on this kernel body. Pass as
#: ``cute_dsl_fmha_paged_prefill(..., **RUBIN_GQA_TUNED_POLICY)``.
RUBIN_GQA_TUNED_POLICY = MappingProxyType(
    {
        "scheduler": "auto",
        "sequence_barrier": False,
        "mma_rowsum": True,
        "packed_nvfp4_epilogue": True,
    }
)


def rubin_paged_prefill_query_layout(
    query_lens: Sequence[int], *, fp4_output: bool = True
) -> tuple[str, str]:
    """Pick ``(query_layout, output_store)`` from host query lengths.

    Measured order for the tuned exports: the uniform layout when two or more
    requests have equal lengths; otherwise packed ragged TMA for NVFP4
    output, or for BF16 output packed dense TMA when every request but the
    last fills whole 256-row Q tiles, else direct stores.
    """
    lens = tuple(int(n) for n in query_lens)
    if not lens or min(lens) <= 0:
        raise ValueError("query_lens must be a nonempty list of positive lengths")
    if len(lens) > 1 and all(n == lens[0] for n in lens):
        return "uniform", "tma"
    if fp4_output:
        return "packed", "ragged_tma"
    if all(n % _GQA_Q_ROWS_PER_CTA == 0 for n in lens[:-1]):
        return "packed", "tma"
    return "packed", "direct"


def _load_kernel(q, varlen=True, *, is_persistent=False, **kwargs):
    try:
        return get_cute_dsl_fmha_kernel(
            _get_gpu_arch(q.device),
            q.dtype,
            q.dtype,
            is_persistent=is_persistent,
            enable_tvm_ffi=True,
            varlen=varlen,
            **kwargs,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        raise RuntimeError(
            "The requested Rubin FMHA AOT specialization is unavailable. It must "
            "be in the pinned cute-dsl FMHA collection (ArtifactPath.DSL_FMHA = "
            f"{DSL_FMHA_ARTIFACT_PATH!r}) or, when FLASHINFER_DSL_FMHA_LOCAL_DIR "
            "is set, in that directory. The MR199 collection (a10580e7) has no "
            "HND/NVFP4, corrected batched-ragged, tuned GQA or persistent ViT "
            "exports. There is no JIT fallback for this entry point."
        ) from exc


# Persistent ViT policy measured on Rubin (212 SMs) for Qwen3-VL (16 heads,
# logical D72) under encoder CUDA graphs: the MMA row-sum exports are 2-3%
# faster than the reference build's policy at lower error, and static
# Work-ID strides beat the dynamic queue up to 8192 tokens of graph capacity
# (the queue wins above, as in the reference build).
RUBIN_VIT_TUNED_POLICY = {"persistent": True, "mma_rowsum": True}
_RUBIN_VIT_STATIC_WORK_MAX_TOKENS = 8192


def rubin_vit_use_work_queue(token_capacity: int) -> bool:
    """Whether a persistent ``cute_dsl_fmha_vit`` launch should pass a counter.

    ``token_capacity`` is the token extent of the Q/K/V buffers (the encoder
    CUDA graph capacity). It is fixed per captured graph, unlike the
    replayed sequence lengths, so the choice is stable under graph replay.
    """
    if not isinstance(token_capacity, int) or token_capacity <= 0:
        raise ValueError("token_capacity must be a positive host integer")
    return token_capacity > _RUBIN_VIT_STATIC_WORK_MAX_TOKENS


def cute_dsl_fmha_vit_work_count(sequence_lengths, num_heads: int) -> int:
    """Return the Work-ID count of a persistent ``cute_dsl_fmha_vit`` launch.

    ``sequence_lengths`` are the host token counts described by
    ``qo_indptr`` (CUDA graph padding entries are 0), and ``num_heads`` is
    the launch's head count. Store the result in ``work_counter[1]``.
    """
    # One Work ID covers 2 x 128 query rows of one (sequence, head).
    q_tile_rows = 256
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    q_tiles = 0
    for length in sequence_lengths:
        length = int(length)
        if length < 0:
            raise ValueError("sequence lengths must be non-negative")
        q_tiles += (length + q_tile_rows - 1) // q_tile_rows
    return num_heads * q_tiles


def cute_dsl_fmha_vit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen: int,
    *,
    logical_head_dim: int = 72,
    sm_scale: float | None = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    scale_o: float = 1.0,
    use_fp16_softmax: bool = True,
    persistent: bool = False,
    work_counter: torch.Tensor | None = None,
    mma_rowsum: bool = False,
) -> None:
    """Noncausal FP8 ViT attention at logical D72, with D80 storage.

    ``q/k/v`` are compact E4M3 tensors [total_tokens, heads, 80]; ``o`` has
    the same shape and BF16 dtype. Only its first 72 channels are defined.
    Input padding must be zero. ``qo_indptr`` contains CUDA int32 cumulative
    TOKEN counts (not cuDNN element offsets), shared by self-attention Q/K/V.
    The default softmax scale uses logical D72, never padded D80.

    Metadata values describe sequence lengths <= ``max_seqlen`` within the
    tensor's token capacity. CUDA graph padding may repeat the final indptr
    (empty requests) and leave an unused tensor tail; only active token rows
    are defined. Value checks are the caller's responsibility; this API
    performs no device-to-host reads.
    Scalars are host-side dequantization scales. Output is modified in place.

    The default export launches one CTA per ``max_seqlen`` tile, sequence slot
    and head; most of them exit at once when ``max_seqlen`` is a CUDA graph
    capacity. ``persistent=True`` selects the persistent export instead: a
    grid of at most one CTA per SM (fixed when the artifact is exported)
    walks only the valid 256-row Q tiles and stores O through ragged TMA.
    With ``work_counter``, those CTAs take Work IDs from a global queue
    rather than a static stride. It is a CUDA int32 tensor
    ``[0, cute_dsl_fmha_vit_work_count(lengths, heads)]`` for the lengths
    in ``qo_indptr``; element 1 must be exact. The kernel resets element 0
    after each launch, so sequential launches on one stream (all encoder
    layers, CUDA graph replays) may share one counter; concurrent launches
    may not. Under CUDA graphs, refresh element 1 with ``qo_indptr``.
    ``mma_rowsum=True`` (persistent only) selects the export that accumulates
    the softmax row sum with an auxiliary MMA; on Rubin it is a few percent
    faster at equal or lower error.
    """
    _require_rubin(q)
    if logical_head_dim != 72:
        raise ValueError("The ViT AOT entry point supports logical D72 / storage D80")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _require_tensor(tensor, name, q.device, torch.float8_e4m3fn, 3)
    _require_tensor(o, "o", q.device, torch.bfloat16, 3, align=32)
    _require_tensor(qo_indptr, "qo_indptr", q.device, torch.int32, 1)
    if q.shape != k.shape or q.shape != v.shape or q.shape != o.shape:
        raise ValueError("ViT q/k/v/o must have identical padded shapes")
    total, heads, storage = q.shape
    if storage != 80 or total <= 0 or heads <= 0 or qo_indptr.numel() < 2:
        raise ValueError(
            "ViT requires positive tokens/heads, D80, and a nonempty batch"
        )
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive host integer")
    if mma_rowsum and not persistent:
        raise ValueError("mma_rowsum requires persistent=True")
    if work_counter is not None:
        if not persistent:
            raise ValueError("work_counter requires persistent=True")
        _require_tensor(work_counter, "work_counter", q.device, torch.int32, 1, align=4)
        if work_counter.numel() != 2:
            raise ValueError("work_counter must hold [next_work_idx, total_work]")
    kernel = _load_kernel(
        q,
        is_persistent=persistent,
        out_dtype=o.dtype,
        head_dim=logical_head_dim,
        is_causal=False,
        use_fp16_softmax=use_fp16_softmax,
        # Persistent ViT exports use the ragged-TMA output store, which the
        # dynamic Work-ID queue requires.
        use_ragged_tma=persistent,
        use_dynamic_work_scheduler=work_counter is not None,
        use_mma_rowsum=mma_rowsum,
    )
    scale_log2, scale_sm, scale_out = _scales(
        scale_q, scale_k, scale_v, scale_o, sm_scale, logical_head_dim
    )
    shape = (1, total, heads, 1, storage)
    problem = (
        qo_indptr.numel() - 1,
        max_seqlen,
        total,
        max_seqlen,
        heads,
        heads,
        logical_head_dim,
        logical_head_dim,
    )
    kernel(
        q.view(shape),
        k.view(shape),
        v.view(shape),
        o.view(shape),
        problem,
        qo_indptr,
        qo_indptr,
        None,
        None,
        scale_log2,
        scale_sm,
        scale_out,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        None,
        work_counter,
    )


def cute_dsl_fmha_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    *,
    kv_layout: str = "HND",
    sm_scale: float | None = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    scale_o: float = 1.0,
    output_block_scale: torch.Tensor | None = None,
    o_sf_scale: float | None = None,
    o_sf_start_index: int = 0,
    use_fp16_softmax: bool = True,
    output_store: str | None = None,
    query_layout: str = "packed",
    scheduler: str | None = None,
    sequence_barrier: bool = True,
    rescale_threshold: float | None = None,
    mma_rowsum: bool = False,
    packed_nvfp4_epilogue: bool = False,
) -> None:
    """Causal Rubin paged FP8 GQA, optionally with fused NVFP4 output.

    Q is compact E4M3 [total_q, Hq, 128]. K/V are E4M3 pools of shape
    [pages, Hkv, page_size, 128] (HND) or compact
    [pages, page_size, Hkv, 128] (NHD). HND accepts non-overlapping positive
    page/head/token strides, including views into interleaved K/V storage.
    Both pools must have identical shapes/strides. No cache repacking occurs.

    ``qo_indptr`` and ``kv_indptr`` contain cumulative TOKEN counts, NOT page
    counts. ``kv_lens`` has the corresponding per-request KV token counts.
    All metadata is CUDA int32; page_table has one compact row per request.
    Caller guarantees valid page IDs, positive lengths, matching cumulative
    counts and max-length bounds. There are no synchronizing value checks.

    BF16 output has shape [total_q, Hq, 128]. Fused NVFP4 output uses uint8
    packed E2M1 storage [total_q, 64, 64] and requires ``output_block_scale``:
    the FULL contiguous E4M3 Layout128x4 allocation, with exactly 512 columns
    and rows rounded up to 128. ``o_sf_start_index`` selects the first row;
    pass an output payload slice but never a row-sliced scale buffer.
    ``o_sf_scale`` is the positive global encoding multiplier. ``scale_o``
    remains 1 for NVFP4; it is not the FP4 global scale.
    ``output_store=None`` selects ragged TMA for NVFP4 and direct stores for
    BF16. NVFP4 also accepts ``output_store="direct"`` for comparison.
    ``output_store="tma"`` stores packed O through a dense TMA descriptor; the
    caller guarantees that every request except the last has a query length
    that is a multiple of 256 (one CTA's Q tile), otherwise a tile spanning two
    requests overwrites the next request's rows.

    ``query_layout="uniform"`` views Q/O as ``[batch, max_q_len, ...]``
    without ``qo_indptr`` (every request has ``max_q_len`` query tokens;
    checked as ``batch * max_q_len == total_q``) and always stores O through
    TMA. :func:`rubin_paged_prefill_query_layout` picks the layout and store
    from host query lengths.

    ``scheduler``, ``sequence_barrier``, ``rescale_threshold``,
    ``mma_rowsum`` and ``packed_nvfp4_epilogue`` select exports built with
    those kernel options (names ``_cluster``/``_clc``, ``_nosb``,
    ``_rt<threshold>``, ``_rowsum``, ``_sfpack``). ``scheduler=None`` is the
    legacy launch of the original exports; ``"cluster"`` the static scheduler
    with cluster launch; ``"clc"`` Rubin hardware CLC Work IDs (page size 16
    with a TMA output store); ``"auto"`` is ``"clc"`` where allowed and
    ``"cluster"`` otherwise. ``packed_nvfp4_epilogue`` applies to NVFP4 output
    only and is ignored for BF16. :data:`RUBIN_GQA_TUNED_POLICY` holds the
    measured-best settings. The defaults load the same exports as before.

    Warm up this explicit AOT-only path before CUDA graph capture. Missing
    specializations raise instead of falling back to BF16 or public JIT.
    """
    _require_rubin(q)
    _require_tensor(q, "q", q.device, torch.float8_e4m3fn, 3)
    for name, tensor in (("k_cache", k_cache), ("v_cache", v_cache)):
        _require_tensor(tensor, name, q.device, q.dtype, 4, compact=False, align=32)
    if kv_layout not in ("HND", "NHD"):
        raise ValueError("kv_layout must be HND or NHD")
    if k_cache.shape != v_cache.shape or k_cache.stride() != v_cache.stride():
        raise ValueError("K/V cache shapes and strides must match")
    if kv_layout == "HND":
        pages, hk, page_size, d = k_cache.shape
        sp, sh, sn, sd = k_cache.stride()
        if sd != 1 or sn < d or sh < page_size * sn or sp < hk * sh:
            raise ValueError(
                "HND cache requires non-overlapping positive H/N/D strides"
            )
        if any(stride % 16 for stride in (sp, sh, sn)):
            raise ValueError("HND page/head/token strides must be multiples of 16")
    else:
        pages, page_size, hk, d = k_cache.shape
        if not k_cache.is_contiguous() or not v_cache.is_contiguous():
            raise ValueError("The NHD export requires compact page pools")
    total, hq, qd = q.shape
    if (qd, d) != (128, 128) or page_size not in (16, 64, 128):
        raise ValueError("Paged GQA requires D128 and page size 16, 64 or 128")
    if min(total, hq, hk, pages) <= 0 or hq % hk:
        raise ValueError(
            "Paged GQA requires positive dimensions and Hq divisible by Hkv"
        )
    for name, tensor, ndim in (
        ("qo_indptr", qo_indptr, 1),
        ("kv_indptr", kv_indptr, 1),
        ("page_table", page_table, 2),
        ("kv_lens", kv_lens, 1),
    ):
        _require_tensor(
            tensor,
            name,
            q.device,
            torch.int32,
            ndim,
            align=4 if name == "kv_lens" else 16,
        )
    batch = page_table.shape[0]
    if batch <= 0 or qo_indptr.numel() != batch + 1 or kv_indptr.numel() != batch + 1:
        raise ValueError("Cumulative token counts must have batch_size + 1 entries")
    if kv_lens.numel() != batch:
        raise ValueError("kv_lens must have one entry per request")
    if (
        batch > 1
        and kv_layout == "NHD"
        and DSL_FMHA_ARTIFACT_PATH.startswith("a10580e7")
        and not os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR")
    ):
        raise RuntimeError(
            "MR199's NHD paged export ties page-table batch to packed Q batch=1. "
            "Multi-sequence ragged GQA requires the corrected exporter build in "
            "FLASHINFER_DSL_FMHA_LOCAL_DIR until a fixed collection is published."
        )
    if any(not isinstance(n, int) or n <= 0 for n in (max_q_len, max_kv_len)):
        raise ValueError("max_q_len and max_kv_len must be positive host integers")
    if page_table.shape[1] * page_size < max_kv_len:
        raise ValueError("page_table has insufficient columns for max_kv_len")

    fp4 = o.dtype == torch.uint8
    if query_layout not in ("packed", "uniform"):
        raise ValueError("query_layout must be 'packed' or 'uniform'")
    uniform = query_layout == "uniform"
    if output_store is None:
        output_store = "tma" if uniform else "ragged_tma" if fp4 else "direct"
    if output_store not in ("direct", "ragged_tma", "tma") or (
        not fp4 and output_store == "ragged_tma"
    ):
        raise ValueError(
            "output_store supports direct or tma, or ragged_tma for NVFP4 only"
        )
    if uniform:
        if output_store != "tma":
            raise ValueError("The uniform query layout stores O through TMA only")
        # Every length is <= max_q_len, so the sum equals batch * max_q_len
        # exactly when all requests have max_q_len query tokens.
        if batch * max_q_len != total:
            raise ValueError(
                "The uniform query layout requires equal query lengths "
                "(batch * max_q_len == total_q)"
            )
    if scheduler == "auto":
        scheduler = "clc" if output_store != "direct" and page_size == 16 else "cluster"
    if scheduler not in (None, "cluster", "clc"):
        raise ValueError("scheduler must be None, 'auto', 'cluster' or 'clc'")
    if scheduler == "clc" and (output_store == "direct" or page_size != 16):
        raise ValueError("CLC scheduling requires page size 16 and a TMA output store")
    if rescale_threshold is not None:
        rescale_threshold = float(rescale_threshold)
        if not math.isfinite(rescale_threshold) or rescale_threshold < 0:
            raise ValueError("rescale_threshold must be finite and non-negative")
    _require_tensor(
        o, "o", q.device, torch.uint8 if fp4 else torch.bfloat16, 3, align=32
    )
    if o.shape != (total, hq, 64 if fp4 else 128):
        raise ValueError(
            "Output shape does not match the BF16 or packed NVFP4 contract"
        )
    sf = sf_scale = sf_start = None
    if fp4:
        if hq != 64:
            raise ValueError("The NVFP4 export is specialized for Hq=64")
        if output_block_scale is None:
            raise ValueError("Packed NVFP4 output requires output_block_scale")
        _require_tensor(output_block_scale, "output_block_scale", q.device, q.dtype, 2)
        if not isinstance(o_sf_start_index, int) or o_sf_start_index < 0:
            raise ValueError("o_sf_start_index must be a nonnegative host integer")
        scale_rows = ((o_sf_start_index + total + 127) // 128) * 128
        scale_cols = ((hq * d // 16 + 3) // 4) * 4
        if (
            output_block_scale.shape[0] < scale_rows
            or output_block_scale.shape[1] != scale_cols
        ):
            raise ValueError(
                "The full Layout128x4 scale allocation has the wrong shape"
            )
        if scale_o != 1.0:
            raise ValueError("Use o_sf_scale, not scale_o, for NVFP4 encoding")
        sf = output_block_scale.view(-1)
        sf_scale = Float32(_positive_scalar(o_sf_scale, "o_sf_scale"))
        sf_start = Int32(o_sf_start_index)
        packed_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if packed_dtype is None:
            raise RuntimeError(
                "NVFP4 AOT requires PyTorch float4_e2m1fn_x2 for the TVM-FFI ABI"
            )
        # DLPack sees float4_e2m1fnx2; the CuTe FFI adapter expands logical D.
        # Passing ordinary uint8 would fail the exported Float4 tensor schema.
        o_arg = o.view(packed_dtype)
    else:
        if output_block_scale is not None or o_sf_scale is not None or o_sf_start_index:
            raise ValueError("Output scale arguments apply only to packed NVFP4 output")
        o_arg = o

    kernel = _load_kernel(
        q,
        varlen=not uniform,
        out_dtype=o.dtype,
        head_dim=d,
        is_causal=True,
        use_pdl=True,
        page_size=page_size,
        kv_layout=kv_layout,
        num_q_heads=hq if fp4 else None,
        use_fp16_softmax=use_fp16_softmax,
        use_ragged_tma=output_store == "ragged_tma",
        use_dense_tma=output_store == "tma" and not uniform,
        scheduler=scheduler,
        disable_sequence_barrier=not sequence_barrier,
        rescale_threshold=rescale_threshold,
        use_mma_rowsum=mma_rowsum,
        use_packed_nvfp4_epilogue=packed_nvfp4_epilogue and fp4,
    )
    scale_log2, scale_sm, scale_out = _scales(
        scale_q, scale_k, scale_v, scale_o, sm_scale, d
    )
    if kv_layout == "HND":
        k_arg, v_arg = k_cache, v_cache
    else:
        k_arg = k_cache.view(pages, page_size, hk, 1, d)
        v_arg = v_cache.view(pages, page_size, hk, 1, d)
    if uniform:
        # Fixed-length Q: [batch, max_q_len] rows, no Q cumulative counts.
        q_batch, q_extent, q_indptr = batch, max_q_len, None
    else:
        q_batch, q_extent, q_indptr = 1, total, qo_indptr
    problem = (batch, max_q_len, q_extent, max_kv_len, hq, hk, d, d)
    kernel(
        q.view(q_batch, q_extent, hk, hq // hk, d),
        k_arg,
        v_arg,
        o_arg.view(q_batch, q_extent, hk, hq // hk, o.shape[-1]),
        problem,
        q_indptr,
        kv_indptr,
        None,
        None,
        scale_log2,
        scale_sm,
        scale_out,
        None,
        None,
        Int32(0),
        None,
        None,
        True,
        page_table,
        sf,
        sf_scale,
        sf_start,
        None,
    )
