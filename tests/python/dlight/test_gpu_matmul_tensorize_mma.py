# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring
import pytest

import tvm.testing
from tvm import dlight as dl
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    @pytest.fixture
    def transform(self):
        def transform(mod):
            with Target("nvidia/nvidia-a100"):
                return dl.ApplyDefaultSchedule(dl.gpu.MatmulTensorizationMMA())(mod)

        return transform


class TestNTMatmulMixedPrecision(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(128), T.int64(128)), "float16")
        B = T.match_buffer(p_B, (T.int64(128), T.int64(128)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(128), T.int64(128)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(128), T.int64(128)))
        for i0, i1, i2, k in T.grid(b, T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_i2, v_k])
        for i0, i1, i2 in T.grid(b, T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def expected(X: T.Buffer((256, 256), "float16"), W: T.Buffer((256, 256), "float16"), compute: T.Buffer((256, 256), "float16")):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        X_reindex_shared_dyn = T.alloc_buffer((1, 256, 256), "float16", scope="shared.dyn")
        W_reindex_shared_dyn = T.alloc_buffer((1, 256, 256), "float16", scope="shared.dyn")
        X_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((1, 256, 256), "float16", scope="wmma.matrix_a")
        W_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((1, 256, 256), "float16", scope="wmma.matrix_b")
        compute_reindex_shared_dyn = T.alloc_buffer((1, 256, 256), "float16", scope="shared.dyn")
        compute_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((1, 256, 256), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(2, thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(16, thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(2, 2):
                            with T.block("compute_o_init"):
                                v0_o = T.axis.spatial(1, ax0)
                                v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax1_0_3_init)
                                v2_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax2_0_3_init)
                                T.reads()
                                T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                with T.block("compute_init_o"):
                                    v1_i_init_o = T.axis.spatial(1, 0)
                                    v2_i_init_o = T.axis.spatial(1, 0)
                                    T.reads()
                                    T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C = T.match_buffer(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.float32(0))
                        for ax3_0_0 in range(4):
                            for ax0_ax1_fused_0 in range(4):
                                for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("X_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(256, ax1_0_0_ax2_0_0_fused * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 64)
                                                v2 = T.axis.spatial(256, ax3_0_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 64)
                                                T.reads(X[v1, v2])
                                                T.writes(X_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                                X_reindex_shared_dyn[v0, v1, v2] = X[v1, v2]
                            for ax0_ax1_fused_0 in range(4):
                                for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("W_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(256, ax1_0_1_ax2_0_1_fused * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 64)
                                                v2 = T.axis.spatial(256, ax3_0_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 64)
                                                T.reads(W[v1, v2])
                                                T.writes(W_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                                W_reindex_shared_dyn[v0, v1, v2] = W[v1, v2]
                            for ax3_0_1 in range(4):
                                for ax0_0 in T.unroll(2):
                                    for ax1_0 in T.unroll(1):
                                        with T.block("X_reindex_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(16, ax3_0_0 * 4 + ax3_0_1 + ax1_0)
                                            T.reads(X_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(X_reindex_shared_dyn_wmma_matrix_a[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A = T.match_buffer(X_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(X_reindex_shared_dyn_wmma_matrix_a[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(2):
                                    for ax1_0 in T.unroll(1):
                                        with T.block("W_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(16, ax3_0_0 * 4 + ax3_0_1 + ax1_0)
                                            T.reads(W_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(W_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A = T.match_buffer(W_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(W_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(2, 2):
                                    with T.block("compute_o_update"):
                                        v0_o = T.axis.spatial(1, ax0)
                                        v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax1_0_3)
                                        v2_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax2_0_3)
                                        v3_o = T.axis.reduce(16, ax3_0_0 * 4 + ax3_0_1)
                                        T.reads(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], X_reindex_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], W_reindex_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                        with T.block("compute_o"):
                                            v1_i_o = T.axis.spatial(1, 0)
                                            v2_i_o = T.axis.spatial(1, 0)
                                            v3_i_o = T.axis.reduce(1, 0)
                                            T.reads(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], X_reindex_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], W_reindex_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                            T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A = T.match_buffer(X_reindex_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(W_reindex_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, A.data, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, B.data, B.elem_offset // B.strides[0] // 16 * (B.strides[0] // 16) + B.elem_offset % B.strides[0] // 16, C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16)
                        for ax0_0, ax1_0 in T.grid(2, 2):
                            with T.block("compute_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(1, 0)
                                v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax0_0)
                                v2_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax1_0)
                                T.reads(compute_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                T.writes(compute_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                A = T.match_buffer(compute_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(compute_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * 16, 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(8):
                            for ax0_ax1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("compute_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(256, ax1_0_0_ax2_0_0_fused * 128 + ax2_0_2_ax1_0_2_fused % 4 * 32 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 32)
                                        v2 = T.axis.spatial(256, ax1_0_1_ax2_0_1_fused * 128 + ax2_0_2_ax1_0_2_fused // 4 * 32 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 32)
                                        T.reads(compute_reindex_shared_dyn[v0, v1, v2])
                                        T.writes(compute[v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        compute[v1, v2] = compute_reindex_shared_dyn[v0, v1, v2]

    # fmt: on



class TestTNMatmulMixedPrecision(BaseBeforeAfter):
    # fmt: off
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(128), T.int64(128)), "float16")
        B = T.match_buffer(p_B, (T.int64(128), T.int64(128)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(128), T.int64(128)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(128), T.int64(128)))
        for i0, i1, i2, k in T.grid(b, T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", A[v_i0, v_k, v_i1]) * T.Cast("float32", B[v_k, v_i2])
        for i0, i1, i2 in T.grid(b, T.int64(128), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func
    def expected(X: T.Buffer((256, 256), "float16"), W: T.Buffer((256, 256), "float16"), compute: T.Buffer((256, 256), "float16")):
        T.func_attr({"global_symbol": "main", "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        X_reindex_shared_dyn = T.alloc_buffer((1, 256, 256), "float16", scope="shared.dyn")
        W_reindex_shared_dyn = T.alloc_buffer((1, 256, 256), "float16", scope="shared.dyn")
        X_reindex_shared_dyn_wmma_matrix_a = T.alloc_buffer((1, 256, 256), "float16", scope="wmma.matrix_a")
        W_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((1, 256, 256), "float16", scope="wmma.matrix_b")
        compute_reindex_shared_dyn = T.alloc_buffer((1, 256, 256), "float16", scope="shared.dyn")
        compute_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((1, 256, 256), "float16", scope="wmma.accumulator")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding(2, thread="blockIdx.x"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(2, thread="blockIdx.y"):
                    for ax2_0_2_ax1_0_2_fused in T.thread_binding(16, thread="threadIdx.y"):
                        for ax1_0_3_init, ax2_0_3_init in T.grid(2, 2):
                            with T.block("compute_o_init"):
                                v0_o = T.axis.spatial(1, ax0)
                                v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax1_0_3_init)
                                v2_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax2_0_3_init)
                                T.reads()
                                T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                with T.block("compute_init_o"):
                                    v1_i_init_o = T.axis.spatial(1, 0)
                                    v2_i_init_o = T.axis.spatial(1, 0)
                                    T.reads()
                                    T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    C = T.match_buffer(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.float32(0))
                        for ax3_0_0 in range(4):
                            for ax0_ax1_fused_0 in range(4):
                                for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("X_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(256, ax1_0_0_ax2_0_0_fused * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 64)
                                                v2 = T.axis.spatial(256, ax3_0_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 64)
                                                T.reads(X[v1, v2])
                                                T.writes(X_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                                X_reindex_shared_dyn[v0, v1, v2] = X[v1, v2]
                            for ax0_ax1_fused_0 in range(4):
                                for ax0_ax1_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(4):
                                            with T.block("W_reindex_shared.dyn"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial(256, ax1_0_1_ax2_0_1_fused * 128 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 64)
                                                v2 = T.axis.spatial(256, ax3_0_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 64)
                                                T.reads(W[v1, v2])
                                                T.writes(W_reindex_shared_dyn[v0, v1, v2])
                                                T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                                W_reindex_shared_dyn[v0, v1, v2] = W[v1, v2]
                            for ax3_0_1 in range(4):
                                for ax0_0 in T.unroll(2):
                                    for ax1_0 in T.unroll(1):
                                        with T.block("X_reindex_shared.dyn_wmma.matrix_a_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(16, ax3_0_0 * 4 + ax3_0_1 + ax1_0)
                                            T.reads(X_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(X_reindex_shared_dyn_wmma_matrix_a[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A = T.match_buffer(X_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(X_reindex_shared_dyn_wmma_matrix_a[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "row_major")
                                for ax0_0 in T.unroll(2):
                                    for ax1_0 in T.unroll(1):
                                        with T.block("W_reindex_shared.dyn_wmma.matrix_b_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(16, ax3_0_0 * 4 + ax3_0_1 + ax1_0)
                                            T.reads(W_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(W_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A = T.match_buffer(W_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                            C = T.match_buffer(W_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "col_major")
                                for ax1_0_3, ax2_0_3 in T.grid(2, 2):
                                    with T.block("compute_o_update"):
                                        v0_o = T.axis.spatial(1, ax0)
                                        v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax1_0_3)
                                        v2_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax2_0_3)
                                        v3_o = T.axis.reduce(16, ax3_0_0 * 4 + ax3_0_1)
                                        T.reads(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], X_reindex_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], W_reindex_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                        with T.block("compute_o"):
                                            v1_i_o = T.axis.spatial(1, 0)
                                            v2_i_o = T.axis.spatial(1, 0)
                                            v3_i_o = T.axis.reduce(1, 0)
                                            T.reads(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], X_reindex_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], W_reindex_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                            T.writes(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            A = T.match_buffer(X_reindex_shared_dyn_wmma_matrix_a[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                            B = T.match_buffer(W_reindex_shared_dyn_wmma_matrix_b[0, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], (16, 16), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                            C = T.match_buffer(compute_reindex_shared_dyn_wmma_accumulator[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                            T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, A.data, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, B.data, B.elem_offset // B.strides[0] // 16 * (B.strides[0] // 16) + B.elem_offset % B.strides[0] // 16, C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16)
                        for ax0_0, ax1_0 in T.grid(2, 2):
                            with T.block("compute_reindex_shared.dyn_wmma.accumulator_o"):
                                v0_o = T.axis.spatial(1, 0)
                                v1_o = T.axis.spatial(16, ax1_0_0_ax2_0_0_fused * 8 + ax2_0_2_ax1_0_2_fused % 4 * 2 + ax0_0)
                                v2_o = T.axis.spatial(16, ax1_0_1_ax2_0_1_fused * 8 + ax2_0_2_ax1_0_2_fused // 4 * 2 + ax1_0)
                                T.reads(compute_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                T.writes(compute_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                A = T.match_buffer(compute_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                C = T.match_buffer(compute_reindex_shared_dyn[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * 16, 2), C.strides[0], "row_major")
                        for ax0_ax1_fused_0 in range(8):
                            for ax0_ax1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("compute_reindex_shared.dyn"):
                                        v0 = T.axis.spatial(1, 0)
                                        v1 = T.axis.spatial(256, ax1_0_0_ax2_0_0_fused * 128 + ax2_0_2_ax1_0_2_fused % 4 * 32 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 32)
                                        v2 = T.axis.spatial(256, ax1_0_1_ax2_0_1_fused * 128 + ax2_0_2_ax1_0_2_fused // 4 * 32 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 32)
                                        T.reads(compute_reindex_shared_dyn[v0, v1, v2])
                                        T.writes(compute[v1, v2])
                                        T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                        compute[v1, v2] = compute_reindex_shared_dyn[v0, v1, v2]

    # fmt: on




if __name__ == "__main__":
    tvm.testing.main()
