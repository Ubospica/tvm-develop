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
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Tuple

from torch import dequantize

from tvm import DataType, tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV
from tvm.tir.stmt import ForKind
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_f16_A_TRANS_DYN_INTRIN,
    LDMATRIX_f16_B_TRANS_DYN_INTRIN,
    MMA_f16f16f32_TRANS_A_INTRIN,
    MMA_f16f16f32_TRANS_A_TRANS_B_INTRIN,
    MMA_f16f16f32_TRANS_B_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_f32_shared_dyn_INTRIN,
    MMA_store_16x16_f32_shared_dyn_INTRIN_SIMPLE,
    shared_16x16_to_ldmatrix_32x8_layout,
)

from . import utils
from ..base import ScheduleRule, analysis


def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
    skip_blocks: Optional[List[tir.schedule.BlockRV]] = None,
):
    skip_blocks = skip_blocks or []
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            if any(sch.get(producer) == sch.get(skip_block) for skip_block in skip_blocks):
                continue
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)

    msg = "There are some consumers of the cache-write stage that are not properly inlined."
    assert len(sch.get_consumers(block)) == 0, msg


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return matmul_index_map, A_index_map, B_index_map, C_index_map


def get_reduction_blocks(sch, blocks) -> Optional[List[BlockRV]]:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def get_dequantize_block(sch, blocks) -> Optional[BlockRV]:
    # check at least two input and one output
    # at lease one input has uint dtype, and the output dtype is float
    def is_dequantize(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        if len(block_stmt.reads) < 2:
            return False
        has_uint_input = any("uint" in str(region.buffer.dtype) for region in block_stmt.reads)
        if not has_uint_input:
            return False
        if len(block_stmt.writes) != 1 or "float" not in str(block_stmt.writes[0].buffer.dtype):
            return False
        return True

    dequantize_blocks = [block for block in blocks if is_dequantize(block)]
    return dequantize_blocks[0] if len(dequantize_blocks) == 1 else None


def is_identity_or_transpose_block(block_stmt: tir.Block) -> bool:
    iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
    if iter_types != {IterVar.DataPar}:
        return False, False
    if not isinstance(block_stmt.body, tir.BufferStore):
        return False, False
    if not isinstance(block_stmt.body.value, tir.BufferLoad):
        return False, False

    def get_access_vars(region: List[Range]) -> List[Var]:
        axes: List[Var] = []
        for r in region:
            if not _is_one(r.extent):
                return None
            axes.extend(undefined_vars(r.min))
        # remove trivial axis
        trivial_vars = set(
            iter_var.var for iter_var in block_stmt.iter_vars if _is_one(iter_var.dom.extent)
        )
        axes = [axis for axis in axes if axis not in trivial_vars]
        # remove duplicate axis
        axes = [var for i, var in enumerate(axes) if i == 0 or var != axes[i - 1]]
        return axes

    lhs_access_vars = get_access_vars(block_stmt.reads[0].region)[-2:]
    rhs_access_vars = get_access_vars(block_stmt.writes[0].region)[-2:]
    is_identity = list(lhs_access_vars) == list(rhs_access_vars)
    is_transpose = list(lhs_access_vars) != list(rhs_access_vars) and set(lhs_access_vars) == set(
        rhs_access_vars
    )
    return is_identity, is_transpose


def is_identity_block(block_stmt: tir.Block) -> bool:
    return is_identity_or_transpose_block(block_stmt)[0]


def is_transpose_block(block_stmt: tir.Block) -> bool:
    return is_identity_or_transpose_block(block_stmt)[1]


def inline_transpose_block(sch: tir.Schedule, blocks: List[tir.schedule.BlockRV]):
    result_blocks = []
    for block in blocks:
        if not is_transpose_block(sch.get(block)):
            result_blocks.append(block)
            continue
        try:
            sch.compute_inline(block)
        except:
            try:
                sch.reverse_compute_inline(block)
            except:
                result_blocks.append(block)
    return result_blocks


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


class MatmulTensorizationMMA(ScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        # We first inline all transpose blocks for later analysis of transposed A and B
        blocks = inline_transpose_block(sch, blocks)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        dequantize_block = get_dequantize_block(sch, blocks)

        main_block = reduction_blocks[0]
        main_block_stmt = sch.get(main_block)

        # Supported data types:
        # fp16, fp16, fp16: fp16 precision
        # fp16, fp16, fp32: fp16 mixed precision
        dtype_a = main_block_stmt.reads[0].buffer.dtype
        dtype_b = main_block_stmt.reads[1].buffer.dtype
        dtype_c = main_block_stmt.writes[0].buffer.dtype
        if dtype_a != dtype_b:
            return None

        # Get index maps
        index_maps = get_index_map(main_block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tile size
        block_m, block_n, block_k = 128, 128, 32

        # tensor core intrinsic size
        micro_size_m, micro_size_n, micro_size_k = 16, 16, 16

        # thread size
        # thread_x == warp_size
        thread_z, thread_y, thread_x = 2, 2, 32

        vector_size = 8
        unroll_depth = 4

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        is_transpose_a = is_transpose_block(sch.get(block))
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        is_transpose_b = is_identity_block(sch.get(block))
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        batch, i, j, k = sch.get_loops(main_block)

        swizzle_factor_for_l2_m = [1, None]
        swizzle_factor_for_l2_n = [1, None]
        # swizzle_factor_for_l2_m = [4, None]
        # swizzle_factor_for_l2_n = [4, None]

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                swizzle_factor_for_l2_m[0] * block_m,
                swizzle_factor_for_l2_n[0] * block_n,
                block_k,
            ],
        )

        # Step 3. Reorder loops for tiling

        # Step 3.1 inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_m])
        j, j_inner = sch.split(j, factors=[None, micro_size_n])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = main_block
        block_outer = sch.blockize(i_inner)

        # Step 3.2 outer loops for tiling
        # split factors for i, j, and k
        micro_block_cnt_in_warp_m = block_m // thread_z // micro_size_m
        micro_block_cnt_in_warp_n = block_n // thread_y // micro_size_n
        micro_block_cnt_in_warp_k = block_k // micro_size_k

        i_factors = swizzle_factor_for_l2_m + [thread_z, micro_block_cnt_in_warp_m]
        j_factors = swizzle_factor_for_l2_n + [thread_y, micro_block_cnt_in_warp_n]
        k_factors = [None, micro_block_cnt_in_warp_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, k0, i2, j2, k1, i3, j3)

        block_axis = sch.fuse(batch, i0, j0, i1, j1)
        sch.bind(block_axis, "blockIdx.x")

        sch.bind(i2, "threadIdx.z")
        sch.bind(j2, "threadIdx.y")

        # Step 4. Read/write to shared mem and register
        def fetch_input(block_outer, read_buffer_idx, tensor_name: Literal["A", "B"], is_transpose):
            # 1) Read to shared memory
            block_read_smem = sch.cache_read(block_outer, read_buffer_idx, "shared.dyn")
            sch.compute_at(block_read_smem, k0)
            auto_inline_producers(
                sch, block_read_smem, [dequantize_block] if dequantize_block else []
            )

            # For transposed read, we directly load transposed tensor from global
            # Then use ldmatrix.trans to handle transpose later
            if (tensor_name == "A" and is_transpose) or (tensor_name == "B" and not is_transpose):
                # specifical handle transpose read (for NN matmul or TT matmul)
                v0, v1 = sch.get_loops(block_read_smem)[-2:]
                sch.reorder(v1, v0)
                sch.transform_layout(block_read_smem, ("write", 0), lambda b, i, j: (b, j, i))

            # bind loops
            fused = sch.fuse(*sch.get_loops(block_read_smem)[-2:])
            f0, f1, f2, f3, f4 = sch.split(fused, [None, thread_z, thread_y, thread_x, vector_size])
            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)

            # swizzling
            sch.annotate(block_read_smem, ann_key="permuted_layout", ann_val=f"g2s_{tensor_name}")

            # 2) Read to register
            block_read_reg = sch.cache_read(block_outer, read_buffer_idx, "warp")
            sch.compute_at(block_read_reg, k1)

            # bind_loops
            micro_size_spatial = micro_size_m if tensor_name == "A" else micro_size_n
            micro_size_1, micro_size_2 = (
                (micro_size_spatial, micro_size_k)
                if not is_transpose
                else (micro_size_k, micro_size_spatial)
            )
            v00, v01 = sch.split(sch.get_loops(block_read_reg)[-2], [None, micro_size_1])
            v10, v11 = sch.split(sch.get_loops(block_read_reg)[-1], [None, micro_size_2])
            sch.reorder(v00, v10, v01, v11)

            # reorder read axis to match the layout of ldmatrix
            sch.transform_layout(
                block_read_reg,
                ("write", 0),
                lambda v0, v1, v2: (
                    v0,
                    v1 // micro_size_1,
                    v2 // micro_size_2,
                    *shared_16x16_to_ldmatrix_32x8_layout(v1 % micro_size_1, v2 % micro_size_2),
                ),
            )

            # swizzling
            mma_read_block = sch.blockize(sch.get_loops(block_read_reg)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=f"s2l_{tensor_name}")

            return block_read_smem, block_read_reg

        block_read_a, block_read_reg_a = fetch_input(block_outer, 0, "A", is_transpose_a)
        block_read_b, block_read_reg_b = fetch_input(block_outer, 1, "B", is_transpose_b)

        # Write to register, and then smem
        def store_output(block_outer, write_buffer_idx):
            # 1) Write to shared memory
            block_write_smem = sch.cache_write(block_outer, write_buffer_idx, "shared.dyn")
            sch.reverse_compute_at(block_write_smem, block_axis)
            auto_inline_consumers(sch, block_write_smem)

            # bind loops
            fused = sch.fuse(*sch.get_loops(block_write_smem)[-2:])
            f0, f1, f2, f3, f4 = sch.split(fused, [None, thread_z, thread_y, thread_x, vector_size])
            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)

            # swizzling
            sch.annotate(block_write_smem, ann_key="permuted_layout", ann_val=f"s2g_C")

            # 2) Write to register
            block_write_reg = sch.cache_write(block_outer, write_buffer_idx, "warp")

            # bind loops
            v0, v1, v2 = sch.get_loops(block_write_reg)[-3:]
            v11, v12, v13 = sch.split(v1, factors=[thread_z, None, micro_size_m])
            v21, v22, v23 = sch.split(v2, factors=[thread_y, None, micro_size_n])
            sch.reorder(v11, v21, v12, v22, v13, v23)
            sch.bind(v11, "threadIdx.z")
            sch.bind(v21, "threadIdx.y")

            # reorder write axis to match the layout of ldmatrix
            sch.transform_layout(
                block_write_reg,
                ("read", 0),
                lambda v0, v1, v2: (
                    v0,
                    v1 // micro_size_m,
                    v2 // micro_size_n,
                    *shared_16x16_to_ldmatrix_32x8_layout(v1 % micro_size_m, v2 % micro_size_n),
                ),
            )

            # swizzling
            mma_read_block = sch.blockize(sch.get_loops(block_write_reg)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=f"l2s_C")

            return block_write_smem, block_write_reg

        block_write_smem, block_write_reg = store_output(block_outer, 0)

        # Step 5. Schedule tensor core computation
        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # unroll k
        # Profiling result shows unrolling k0 is not helpful on A100
        # sch.unroll(k0)
        # k00, k01 = sch.split(k0, factors=[None, 8])
        # sch.unroll(k01)

        # Tensorization by hardware intrinsics
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
        )

        intrin_group = get_mma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype=str(dtype_a),
            out_dtype=str(dtype_c),
            trans_a=is_transpose_a,
            trans_b=is_transpose_b,
        )

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
        sch.tensorize(sch.get_loops(block_read_reg_a)[-2], intrin_group["load_a"])
        sch.tensorize(sch.get_loops(block_read_reg_b)[-2], intrin_group["load_b"])
        sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])
        sch.tensorize(sch.get_loops(block_write_reg)[-2], intrin_group["store"])

        # Step 6. Async pipeline
        sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
        sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

        # Step 7. Handle dequantize block
        # Now we just add a dummy kernel to compute dequantize
        if dequantize_block is not None:
            auto_inline_producers(sch, dequantize_block)
            loops = sch.get_loops(dequantize_block)
            loop = sch.fuse(*loops)
            v0, v1, v2, v3 = sch.split(loop, [None, 128, 2, 4])
            sch.bind(v0, "blockIdx.x")
            sch.bind(v1, "threadIdx.x")
            sch.unroll(v2)
            sch.vectorize(v3)
        return sch


class MatmulTensorization(ScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32
        vector_size = 4

        i_factors, j_factors, k_factors = (
            [None, 1, 4, 2],
            [1, None, 4, 2],
            [None, 4],
        )

        num_ty = i_factors[2] * j_factors[2]
        x_pad_factor = i_factors[2] * i_factors[3]
        y_pad_factor = j_factors[2] * j_factors[3]
        k_pad_factor = k_factors[1]

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = block
        block_outer = sch.blockize(i_inner)

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)
        sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
        sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
        sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
        sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])

        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=8)
            sch.annotate(block_read, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(block_read, "double_buffer_scope", 0)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2)
        b_g2s = fetch_to_shared(block_outer, 1, 2)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype="float16",
            out_dtype="float32",
            trans_b=True,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        # Try to tensorize the init, store and compute block with f16 or f32 intrinsics
        tensorize_success: bool = False

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
            tensorize_success = True
        except:  # pylint: disable=bare-except
            intrin_group = get_wmma_intrin_group(
                load_scope="shared.dyn",
                store_scope="shared.dyn",
                in_dtype="float16",
                out_dtype="float16",
                trans_b=True,
            )

        if not tensorize_success:
            try:
                tensorize_init_store_compute()
                tensorize_success = True
            except:  # pylint: disable=bare-except
                return None
        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch if tensorize_success else None


class Matmul(ScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and "android" in str(target.host):
            return Matmul.Config(
                block_size_x=8,
                block_size_y=8,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=64,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        index_maps = index_maps[0]
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 1. Check Tensor Core support
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent, tir.expr.IntImm):
                    if extent.value <= minimal_tensorize_threshold:
                        apply_tensorization = False
            if apply_tensorization:
                tensorize_sch = MatmulTensorization().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch

        # Step 2. Get schedule config.
        config = self.get_configs(target)

        # Step 3. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(xi, [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumer_chain(sch, l2g)

        remaining_consumers = sch.get_consumers(l2g)

        if len(remaining_consumers) != 0:
            # Some blocks have failed to be inlined to the producer cache-write stage.
            # This could be due to another producer block that has not been scheduled.
            for c in remaining_consumers:
                for p in sch.get_producers(c):
                    if sch.get(p) != sch.get(l2g):
                        sch.compute_inline(p)

            # Try inlining into the cache-write stage again, this time it should succeed.
            auto_inline_consumers(sch, l2g)

        # msg = "There are some consumers of the cache-write stage that are not properly inlined."
        # assert len(sch.get_consumers(l2g)) == 0, msg
        # Step4. Check if there are unbound blocks. Execute fallback scheduling to them.
        def is_scheduled(block: tir.schedule.BlockRV) -> bool:
            loops = sch.get_loops(block)
            loop_kinds = {sch.get(loop).kind for loop in loops}
            return loop_kinds != {ForKind.SERIAL}

        blocks = sch.get_child_blocks(root_block)
        max_threads_per_block = utils.max_threads_per_block(target)
        for block in blocks:
            if is_scheduled(block):
                continue
            # no axis of the block is bound to thread or block
            s_loops = sch.get_loops(block)
            bx, tx = sch.split(  # pylint: disable=invalid-name
                sch.fuse(*s_loops),
                factors=[None, max_threads_per_block],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

        sch.decompose_reduction(main_block, ko)
        return sch
