# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host-only ABI/dispatch tests, runnable without CUDA or PyTorch.

Run directly with Python to bypass the GPU-importing tests/conftest.py. The
actual wrapper function definitions are loaded via AST; only imports and the
tensor/scalar runtime are substituted. This tests metadata wiring, not kernels.
GPU numerical coverage is in test_rubin_fmha_aot.py.
"""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
DEVICE = SimpleNamespace(type="cuda")


class Tensor:
    def __init__(self, shape, dtype="e4m3", *, strides=None, offset=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = DEVICE
        self.ndim = len(shape)
        self.offset = offset
        self._stride = strides or self._compact_stride()

    def _compact_stride(self):
        stride, result = 1, []
        for extent in reversed(self.shape):
            result.insert(0, stride)
            stride *= extent
        return tuple(result)

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self._stride == self._compact_stride()

    def data_ptr(self):
        return 4096 + self.offset * (4 if self.dtype == "int32" else 1)

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self.offset

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], str):
            return Tensor(
                self.shape, shape[0], strides=self._stride, offset=self.offset
            )
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if shape == (-1,):
            shape = (self.numel(),)
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype, offset=self.offset)


TORCH = SimpleNamespace(
    Tensor=Tensor,
    dtype=str,
    device=object,
    float16="fp16",
    bfloat16="bf16",
    float8_e4m3fn="e4m3",
    uint8="uint8",
    int32="int32",
    float4_e2m1fn_x2="e2m1x2",
)


def load_functions(relative_path, names=None):
    tree = ast.parse((ROOT / relative_path).read_text())
    tree.body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and (names is None or node.name in names))
        # Module constants used by the wrappers (whole-module loads only).
        or (names is None and isinstance(node, ast.Assign))
    ]
    ns = {
        "torch": TORCH,
        "math": math,
        "os": os,
        "Float32": float,
        "Int32": int,
        "Optional": __import__("typing").Optional,
        "Sequence": __import__("collections.abc").abc.Sequence,
        "MappingProxyType": __import__("types").MappingProxyType,
        "DSL_FMHA_ARTIFACT_PATH": "a10580e7/fmha/cute-dsl/",
        "_get_gpu_arch": lambda _: "sm_107a",
    }
    exec(compile(tree, str(ROOT / relative_path), "exec"), ns)
    return ns


class RubinFmhaContractTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions("flashinfer/attention/cute_dsl/fmha_rubin.py")
        self.calls = []
        self.selection = []

        def loader(*args, **kwargs):
            self.selection.append((args, kwargs))
            return lambda *values: self.calls.append(values)

        self.ns["get_cute_dsl_fmha_kernel"] = loader

    def paged(self, **kwargs):
        total, batch = 193, 2
        # vLLM split-D cache has a doubled stride between adjacent KV tokens.
        k = Tensor((32, 4, 16, 128), strides=(16384, 4096, 256, 1))
        v = Tensor(k.shape, strides=k.stride())
        inputs = dict(
            q=Tensor((total, 64, 128)),
            k_cache=k,
            v_cache=v,
            o=Tensor((total, 64, 64), "uint8"),
            qo_indptr=Tensor((batch + 1,), "int32"),
            kv_indptr=Tensor((batch + 1,), "int32"),
            page_table=Tensor((batch, 12), "int32"),
            # Deliberately not 16-byte aligned: this is an unused metadata view.
            kv_lens=Tensor((batch,), "int32", offset=1),
            max_q_len=129,
            max_kv_len=177,
            output_block_scale=Tensor((256, 512)),
            o_sf_scale=64.0,
            o_sf_start_index=3,
            scale_q=0.5,
            scale_k=0.25,
            scale_v=2.0,
        )
        inputs.update(kwargs)
        self.ns["cute_dsl_fmha_paged_prefill"](**inputs)

    def test_fp4_abi_and_nonzero_scale_offset(self):
        self.paged()
        args = self.calls[0]
        self.assertEqual(len(args), 23)
        self.assertEqual(args[3].dtype, "e2m1x2")
        self.assertEqual(args[3].shape, (1, 193, 4, 16, 64))
        self.assertEqual(args[4], (2, 129, 193, 177, 64, 4, 128, 128))
        self.assertEqual(args[18].shape, (2, 12))
        self.assertEqual(args[19].shape, (256 * 512,))
        self.assertEqual(args[20:23], (64.0, 3, None))
        self.assertAlmostEqual(args[10], 0.5 * 0.25 / math.sqrt(128))
        self.assertEqual(args[11], 2.0)
        self.assertEqual(self.selection[0][1]["num_q_heads"], 64)
        self.assertTrue(self.selection[0][1]["use_ragged_tma"])
        self.assertEqual(args[1].stride(), (16384, 4096, 256, 1))

    def test_default_paged_selection_is_unchanged(self):
        self.paged()
        selection = self.selection[0][1]
        self.assertTrue(selection["varlen"])
        self.assertIsNone(selection["scheduler"])
        self.assertFalse(selection["use_dense_tma"])
        self.assertFalse(selection["disable_sequence_barrier"])
        self.assertIsNone(selection["rescale_threshold"])
        self.assertFalse(selection["use_mma_rowsum"])
        self.assertFalse(selection["use_packed_nvfp4_epilogue"])

    def test_uniform_query_layout_abi(self):
        policy = dict(
            scheduler="auto",
            sequence_barrier=False,
            mma_rowsum=True,
            packed_nvfp4_epilogue=True,
        )
        self.paged(
            q=Tensor((192, 64, 128)),
            o=Tensor((192, 64, 64), "uint8"),
            max_q_len=96,
            query_layout="uniform",
            **policy,
        )
        args = self.calls[0]
        self.assertEqual(len(args), 23)
        self.assertEqual(args[0].shape, (2, 96, 4, 16, 128))
        self.assertEqual(args[3].shape, (2, 96, 4, 16, 64))
        self.assertEqual(args[4], (2, 96, 96, 177, 64, 4, 128, 128))
        self.assertIsNone(args[5])
        self.assertEqual(args[6].shape, (3,))
        selection = self.selection[0][1]
        self.assertFalse(selection["varlen"])
        self.assertEqual(selection["scheduler"], "clc")
        self.assertFalse(selection["use_ragged_tma"])
        self.assertFalse(selection["use_dense_tma"])
        self.assertTrue(selection["disable_sequence_barrier"])
        self.assertIsNone(selection["rescale_threshold"])
        self.assertTrue(selection["use_mma_rowsum"])
        self.assertTrue(selection["use_packed_nvfp4_epilogue"])

    def test_uniform_query_layout_requires_equal_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal query lengths"):
            self.paged(query_layout="uniform")
        self.assertFalse(self.calls)

    def test_policy_scheduler_follows_output_store(self):
        policy = dict(scheduler="auto", sequence_barrier=False, rescale_threshold=0.0)
        for store, scheduler, dense, ragged in (
            ("ragged_tma", "clc", False, True),
            ("tma", "clc", True, False),
            ("direct", "cluster", False, False),
        ):
            self.selection.clear()
            self.paged(output_store=store, **policy)
            selection = self.selection[0][1]
            self.assertTrue(selection["varlen"])
            self.assertEqual(selection["scheduler"], scheduler)
            self.assertEqual(selection["use_dense_tma"], dense)
            self.assertEqual(selection["use_ragged_tma"], ragged)
        with self.assertRaisesRegex(ValueError, "CLC"):
            self.paged(output_store="direct", scheduler="clc")

    def test_bf16_dense_tma_allowed_ragged_rejected(self):
        bf16 = dict(
            o=Tensor((193, 64, 128), "bf16"),
            output_block_scale=None,
            o_sf_scale=None,
            o_sf_start_index=0,
        )
        self.paged(
            output_store="tma", scheduler="auto", packed_nvfp4_epilogue=True, **bf16
        )
        selection = self.selection[0][1]
        self.assertTrue(selection["use_dense_tma"])
        self.assertIsNone(selection["num_q_heads"])
        self.assertEqual(selection["scheduler"], "clc")
        # The packed NVFP4 epilogue does not apply to BF16 output.
        self.assertFalse(selection["use_packed_nvfp4_epilogue"])
        with self.assertRaisesRegex(ValueError, "ragged_tma for NVFP4"):
            self.paged(output_store="ragged_tma", **bf16)

    def test_tuned_policy_selects_tuned_exports(self):
        self.paged(**self.ns["RUBIN_GQA_TUNED_POLICY"])
        selection = self.selection[0][1]
        self.assertEqual(selection["scheduler"], "clc")
        self.assertTrue(selection["use_ragged_tma"])
        self.assertTrue(selection["disable_sequence_barrier"])
        self.assertIsNone(selection["rescale_threshold"])
        self.assertTrue(selection["use_mma_rowsum"])
        self.assertTrue(selection["use_packed_nvfp4_epilogue"])

    def test_query_layout_selection(self):
        pick = self.ns["rubin_paged_prefill_query_layout"]
        self.assertEqual(pick([3109]), ("packed", "ragged_tma"))
        self.assertEqual(pick([256] * 84), ("uniform", "tma"))
        self.assertEqual(pick([2688] * 8), ("uniform", "tma"))
        self.assertEqual(pick([1] * 128), ("uniform", "tma"))
        self.assertEqual(pick([3109, 777, 64]), ("packed", "ragged_tma"))
        self.assertEqual(pick([3109], fp4_output=False), ("packed", "tma"))
        self.assertEqual(pick([512, 1024, 77], fp4_output=False), ("packed", "tma"))
        self.assertEqual(pick([3109, 777], fp4_output=False), ("packed", "direct"))
        with self.assertRaises(ValueError):
            pick([])

    def vit(self, **kwargs):
        q, k, v = [Tensor((193, 16, 80)) for _ in range(3)]
        o, indptr = Tensor(q.shape, "bf16"), Tensor((3,), "int32")
        self.ns["cute_dsl_fmha_vit"](q, k, v, o, indptr, 129, **kwargs)
        return self.calls[-1], self.selection[-1][1]

    def test_vit_uses_logical_dimension_and_token_indptr(self):
        q, k, v = [Tensor((193, 16, 80)) for _ in range(3)]
        o, indptr = Tensor(q.shape, "bf16"), Tensor((3,), "int32")
        self.ns["cute_dsl_fmha_vit"](q, k, v, o, indptr, 129)
        args = self.calls[0]
        self.assertEqual(len(args), 23)
        self.assertEqual(args[4][-2:], (72, 72))
        self.assertEqual(args[0].shape[-1], 80)
        self.assertIs(args[5], indptr)
        self.assertIs(args[6], indptr)
        self.assertAlmostEqual(args[10], 1 / math.sqrt(72))
        self.assertFalse(args[17])

    def test_vit_defaults_to_nonpersistent_export(self):
        args, selected = self.vit()
        self.assertFalse(selected["is_persistent"])
        self.assertFalse(selected["use_ragged_tma"])
        self.assertFalse(selected["use_dynamic_work_scheduler"])
        self.assertIsNone(args[22])

    def test_vit_persistent_static_work_ids(self):
        args, selected = self.vit(persistent=True)
        self.assertEqual(len(args), 23)
        self.assertTrue(selected["is_persistent"])
        self.assertTrue(selected["use_ragged_tma"])
        self.assertFalse(selected["use_dynamic_work_scheduler"])
        self.assertIsNone(args[22])

    def test_vit_dynamic_work_counter_is_last_argument(self):
        counter = Tensor((2,), "int32")
        args, selected = self.vit(persistent=True, work_counter=counter)
        self.assertEqual(len(args), 23)
        self.assertTrue(selected["is_persistent"])
        self.assertTrue(selected["use_dynamic_work_scheduler"])
        self.assertIs(args[22], counter)
        # The scheduler choice does not change the metadata slots.
        self.assertEqual(args[4], (2, 129, 193, 129, 16, 16, 72, 72))
        self.assertIsNone(args[18])

    def test_vit_work_counter_contract(self):
        with self.assertRaisesRegex(ValueError, "requires persistent"):
            self.vit(work_counter=Tensor((2,), "int32"))
        with self.assertRaisesRegex(ValueError, "next_work_idx"):
            self.vit(persistent=True, work_counter=Tensor((3,), "int32"))
        with self.assertRaisesRegex(ValueError, "work_counter must be"):
            self.vit(persistent=True, work_counter=Tensor((2,), "e4m3"))
        self.assertFalse(self.calls)

    def test_vit_work_count_uses_256_row_tiles(self):
        count = self.ns["cute_dsl_fmha_vit_work_count"]
        self.assertEqual(count([256, 0, 257, 1], 16), 16 * (1 + 0 + 2 + 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            count([-1], 16)
        with self.assertRaisesRegex(ValueError, "num_heads"):
            count([1], 0)

    def test_scale_columns_must_be_exact(self):
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            self.paged(output_block_scale=Tensor((256, 516)))

    def test_scale_allocation_covers_offset(self):
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            self.paged(o_sf_start_index=100)

    def test_scale_buffer_can_be_a_full_region_in_a_memory_pool(self):
        self.paged(output_block_scale=Tensor((256, 512), offset=512))
        self.assertEqual(self.calls[0][19].storage_offset(), 512)

    def test_scales_do_not_accept_device_tensors(self):
        with self.assertRaisesRegex(ValueError, "host scalar"):
            self.paged(scale_q=Tensor((1,), "fp32"))

    def test_no_jit_fallback_for_missing_artifact(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("missing")

        self.ns["get_cute_dsl_fmha_kernel"] = missing
        with self.assertRaisesRegex(RuntimeError, "no JIT fallback"):
            self.paged()
        self.assertFalse(self.calls)

    def test_bad_hnd_strides_rejected(self):
        bad = Tensor((32, 4, 16, 128), strides=(16388, 4096, 256, 1))
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            self.paged(k_cache=bad, v_cache=bad)

    def test_known_broken_published_ragged_variant_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            nhd = Tensor((32, 16, 4, 128))
            with self.assertRaisesRegex(RuntimeError, "page-table batch"):
                self.paged(k_cache=nhd, v_cache=nhd, kv_layout="NHD")

    def test_variant_names_match_exporter(self):
        names = load_functions(
            "flashinfer/attention/cute_dsl/fmha.py",
            {"_dtype_to_str", "_get_variant_name"},
        )
        name = names["_get_variant_name"](
            "e4m3",
            "e4m3",
            "uint8",
            128,
            True,
            False,
            varlen=True,
            enable_tvm_ffi=True,
            use_pdl=True,
            page_size=16,
            kv_layout="HND",
            num_q_heads=64,
            use_fp16_softmax=True,
            use_ragged_tma=True,
        )
        self.assertEqual(
            name,
            "cute_dsl_fmha_e4m3_e2m1_h128_causal_nonpersistent_varlen_pdl_paged16_hnd_qh64_raggedtma_fp16sm_tvmffi",
        )
        name = names["_get_variant_name"](
            "e4m3",
            "e4m3",
            "bf16",
            72,
            False,
            False,
            varlen=True,
            enable_tvm_ffi=True,
            use_fp16_softmax=True,
        )
        self.assertEqual(
            name,
            "cute_dsl_fmha_e4m3_bf16_h72_nocausal_nonpersistent_varlen_fp16sm_tvmffi",
        )
        gqa = dict(
            enable_tvm_ffi=True,
            use_pdl=True,
            page_size=16,
            kv_layout="HND",
            use_fp16_softmax=True,
            disable_sequence_barrier=True,
            rescale_threshold=0.0,
        )
        prefix = "cute_dsl_fmha_e4m3_e2m1_h128_causal_nonpersistent"
        for kwargs, suffix in (
            (
                dict(varlen=True, use_ragged_tma=True, scheduler="clc"),
                "_varlen_pdl_paged16_hnd_qh64_raggedtma_clc_nosb_rt0_fp16sm_tvmffi",
            ),
            (
                dict(varlen=True, use_dense_tma=True, scheduler="clc"),
                "_varlen_pdl_paged16_hnd_qh64_densetma_clc_nosb_rt0_fp16sm_tvmffi",
            ),
            (
                dict(varlen=False, scheduler="clc"),
                "_pdl_paged16_hnd_qh64_clc_nosb_rt0_fp16sm_tvmffi",
            ),
            (
                dict(varlen=True, scheduler="cluster"),
                "_varlen_pdl_paged16_hnd_qh64_cluster_nosb_rt0_fp16sm_tvmffi",
            ),
        ):
            name = names["_get_variant_name"](
                "e4m3",
                "e4m3",
                "uint8",
                128,
                True,
                False,
                num_q_heads=64,
                **gqa,
                **kwargs,
            )
            self.assertEqual(name, prefix + suffix)
        name = names["_get_variant_name"](
            "e4m3",
            "e4m3",
            "uint8",
            128,
            True,
            False,
            varlen=True,
            enable_tvm_ffi=True,
            use_pdl=True,
            page_size=16,
            kv_layout="HND",
            num_q_heads=64,
            use_fp16_softmax=True,
            use_ragged_tma=True,
            scheduler="clc",
            disable_sequence_barrier=True,
            use_mma_rowsum=True,
            use_packed_nvfp4_epilogue=True,
        )
        self.assertEqual(
            name,
            prefix
            + "_varlen_pdl_paged16_hnd_qh64_raggedtma_clc_nosb_rowsum_sfpack_fp16sm_tvmffi",
        )
        for dynamic, suffix in ((False, ""), (True, "_dynwork")):
            name = names["_get_variant_name"](
                "e4m3",
                "e4m3",
                "bf16",
                72,
                False,
                True,
                varlen=True,
                enable_tvm_ffi=True,
                use_fp16_softmax=True,
                use_ragged_tma=True,
                use_dynamic_work_scheduler=dynamic,
            )
            self.assertEqual(
                name,
                "cute_dsl_fmha_e4m3_bf16_h72_nocausal_persistent_varlen"
                f"{suffix}_raggedtma_fp16sm_tvmffi",
            )


if __name__ == "__main__":
    unittest.main()
