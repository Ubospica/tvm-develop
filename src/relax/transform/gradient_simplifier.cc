/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/transform/gradient_simplifier.cc
 * \brief Simplify patterns generated by the gradient pass. Only used in gradient.cc.
 * \sa tvm/relax/transform/gradient.cc
 *
 * We will simplify these patterns:
 * 1. Forward is: out = matmul(a, permute_dims(b))
 *    Then backward is:
 *        grad_a = matmul(grad_out, permute_dims(permute_dims(b)))
 *        grad_b = permute_dims(matmul(permute_dims(a), grad_out))
 *    We will simplify it to:
 *        grad_a = matmul(grad_out, b)
 *        grad_b = matmul(permute_dims(grad_out), a)
 * 2. Forward is: out = matmul(permute_dims(a), b)
 *    Then backward is:
 *        grad_a = permute_dims(matmul(grad_out, permute_dims(b)))
 *        grad_b = matmul(permute_dims(permute_dims(a)), grad_out)
 *    We will simplify it to:
 *        grad_a = matmul(b, permute_dims(grad_out))
 *        grad_b = matmul(a, grad_out)
 * 3. Forward is: out = matmul(permute_dims(a), permute_dims(b))
 *    Then backward is:
 *        grad_a = permute_dims(matmul(grad_out, permute_dims(permute_dims(b))))
 *        grad_b = permute_dims(matmul(permute_dims(permute_dims(a)), grad_out))
 *    We will simplify it to:
 *        grad_a = matmul(permute_dims(b), permute_dims(grad_out))
 *        grad_b = matmul(permute_dims(grad_out), permute_dims(a))
 */

#include "gradient_simplifier.h"

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

/*!
 * \brief Simplify patterns generated by the gradient pass. Especially, simplify the matmul
 * patterns.
 */
class GradientSimplifier : private ExprMutator {
 public:
  /*!
   * \brief Collect all variables that needs to be checkpointed, and remove the start_checkpoint
   * and the end_checkpoint bindings.
   *
   * \param func The original function
   * \return The function with all start_checkpoint and end_checkpoint bindings removed, and a
   * VarIdSet containing all checkpointed vars.
   */
  static Function Transform(const Function& func) {
    return RemoveAllUnused(Downcast<Function>(GradientSimplifier().VisitExpr(func)));
  }

 private:
  static bool is_transpose_op(const CallNode* call_node) {
    if (call_node->op != Op::Get("relax.permute_dims")) {
      return false;
    }
    auto sinfo = MatchStructInfo<TensorStructInfo>(call_node->args[0]);
    if (!sinfo) {
      return false;
    }
    auto ndim = sinfo.value()->ndim;
    if (ndim == kUnknownNDim || ndim == 1) {
      return false;
    }
    if (!call_node->attrs.as<PermuteDimsAttrs>()->axes.defined()) {
      return ndim == 2;
    }
    auto axes = call_node->attrs.as<PermuteDimsAttrs>()->axes.value();
    ICHECK(axes.size() == ndim);
    for (int i = 0; i < ndim - 2; ++i) {
      if (axes[i] != i) {
        return false;
      }
    }
    return axes[ndim - 2] == ndim - 1 && axes[ndim - 1] == ndim - 2;
  }

  // Return permute_dims(expr). Generate the axes needed.
  static Expr get_transpose_op(const Expr& expr) {
    auto sinfo = MatchStructInfo<TensorStructInfo>(expr);
    ICHECK(sinfo);
    auto ndim = sinfo.value()->ndim;
    if (ndim == 1) {
      return expr;
    }
    auto axes = Array<Integer>();
    for (int i = 0; i < ndim - 2; ++i) {
      axes.push_back(i);
    }
    axes.push_back(ndim - 1);
    axes.push_back(ndim - 2);
    return permute_dims(expr, axes);
  }

  // If expr is already in the form of permute_dims in previous bindings, return the input of the
  // permute_dims op
  // Else, return permute_dims(expr)
  Expr get_transpose_according_to_ctx(const Expr& expr) {
    if (!expr->IsInstance<VarNode>()) {
      return get_transpose_op(expr);
    }
    auto prev_expr = builder_->LookupBinding(Downcast<Var>(expr));
    if (!prev_expr || !prev_expr->IsInstance<CallNode>()) {
      return get_transpose_op(expr);
    }
    auto prev_call_node = prev_expr.as<CallNode>();
    if (!is_transpose_op(prev_call_node)) {
      return get_transpose_op(expr);
    }
    return prev_call_node->args[0];
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) {
    auto result = ExprMutator::VisitExpr(GetRef<Expr>(call_node));
    auto new_call_node = result.as<CallNode>();
    auto reemit_and_return = [&]() {
      VLOG(0) << result;
      VLOG(0) << result->struct_info_;
      ReEmitBinding(binding, result);
      return;
    };

    if (!is_transpose_op(new_call_node)) {
      return reemit_and_return();
    }

    auto arg = new_call_node->args[0];
    if (!arg->IsInstance<VarNode>()) {
      return reemit_and_return();
    }

    auto prev_expr = builder_->LookupBinding(Downcast<Var>(arg));
    if (!prev_expr || !prev_expr->IsInstance<CallNode>()) {
      return reemit_and_return();
    }

    auto prev_call_node = prev_expr.as<CallNode>();
    if (is_transpose_op(prev_call_node)) {
      // rewrite rule #1: permute_dims(permute_dims(a)) -> a
      VLOG(0) << prev_call_node << " " << prev_call_node->args[0];
      if (prev_call_node->args[0]->IsInstance<VarNode>()) {
        var_remap_[binding->var->vid] = Downcast<Var>(prev_call_node->args[0]);
        return;
      } else {
        return reemit_and_return();
      }
    } else if (prev_call_node->op == Op::Get("relax.matmul")) {
      // rewrite rule #2: permute_dims(matmul(a, b)) -> matmul(permute_dims(b), permute_dims(a))
      // Should "a" or "b" already be in the form of "permute_dims", the redundant permute_dims
      // operation should be eliminated

      // Skip matmuls with 1-dim input because in these cases we cannot simply transpose the input
      auto a_dim = MatchStructInfo<TensorStructInfo>(prev_call_node->args[0]).value()->ndim;
      auto b_dim = MatchStructInfo<TensorStructInfo>(prev_call_node->args[1]).value()->ndim;
      if (a_dim == 1 || b_dim == 1) {
        return reemit_and_return();
      }

      auto a = get_transpose_according_to_ctx(prev_call_node->args[0]);
      auto b = get_transpose_according_to_ctx(prev_call_node->args[1]);
      result =
          ExprMutator::VisitExpr(matmul(b, a, prev_call_node->attrs.as<MatmulAttrs>()->out_dtype));
      ReEmitBinding(binding, result);
      return;
    } else {
      return reemit_and_return();
    }
  }

  // Expr VisitExpr_(const CallNode* call_node) final {
  //   auto result = ExprMutator::VisitExpr_(call_node);
  //   auto new_call_node = result.as<CallNode>();
  //   if (!is_transpose_op(new_call_node)) {
  //     return result;
  //   }
  //   auto arg = new_call_node->args[0];
  //   if (!arg->IsInstance<VarNode>()) {
  //     return result;
  //   }
  //   auto prev_expr = builder_->LookupBinding(Downcast<Var>(arg));
  //   if (!prev_expr || !prev_expr->IsInstance<CallNode>()) {
  //     return result;
  //   }
  //   auto prev_call_node = prev_expr.as<CallNode>();
  //   if (is_transpose_op(prev_call_node)) {
  //     // rewrite rule #1: permute_dims(permute_dims(a)) -> a
  //     VLOG(0) << prev_call_node << " " << prev_call_node->args[0];
  //     return prev_call_node->args[0];
  //   } else if (prev_call_node->op == Op::Get("relax.matmul")) {
  //     // rewrite rule #2: permute_dims(matmul(a, b)) -> matmul(permute_dims(b), permute_dims(a))
  //     // Should "a" or "b" already be in the form of "permute_dims", the redundant permute_dims
  //     // operation should be eliminated

  //     // Skip matmuls with 1-dim input because in these cases we cannot simply transpose the
  //     input auto a_dim =
  //     MatchStructInfo<TensorStructInfo>(prev_call_node->args[0]).value()->ndim; auto b_dim =
  //     MatchStructInfo<TensorStructInfo>(prev_call_node->args[1]).value()->ndim; if (a_dim == 1 ||
  //     b_dim == 1) {
  //       return result;
  //     }
  //     auto a = get_transpose_according_to_ctx(prev_call_node->args[0]);
  //     auto b = get_transpose_according_to_ctx(prev_call_node->args[1]);
  //     return ExprMutator::VisitExpr(
  //         matmul(b, a, prev_call_node->attrs.as<MatmulAttrs>()->out_dtype));
  //   } else {
  //     return result;
  //   }
  // }
};

Function SimplifyGradient(const Function& func) { return GradientSimplifier::Transform(func); }

}  // namespace relax
}  // namespace tvm
