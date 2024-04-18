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
# pylint: disable=invalid-name
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""
from typing import List, Union

from tvm import tir
from tvm.target import Target

from ..base import normalize_prim_func, try_inline_contiguous_spatial
from .base import GPUScheduleRule


class GeneralReduction(GPUScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
        else:
            len_tx = 64
            unroll_depth = 64

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter == 0:
            sch.add_unit_loop(block_infos[-1].block_rv)
            num_last_block_iter = 1
        if num_last_block_iter < len(dom_kind):
            dtype = block_infos[0].iters[0].var.dtype
            index_map = tir.IndexMap.from_func(
                lambda *iters: (
                    [tir.const(0, dtype)] * (len(dom_kind) - num_last_block_iter) + list(iters)
                ),
                ndim=num_last_block_iter,
            )
            print(index_map)
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx = sch.fuse(*loops[:num_leading_s])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, buffer_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="pragma_unroll_explicit", ann_val=1)

        # TODO: It's just a workaround to avoid unroll spatial loops, because of the bug of
        # the pass lower-thread-allreduce. We should fix it in the future.
        # sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        # sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)
        return sch
