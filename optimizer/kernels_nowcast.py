"""Triton kernels for efficient weight nowcasting operations.

Includes fused operations for:
- Weight trajectory encoding with learned projections
- Graph message passing with attention-weighted aggregation
- Batched weight updates with momentum
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable
from caramba.optimizer.runtime import triton_supported

__all__ = [
    "fused_weight_encode",
    "fused_graph_message_pass",
    "fused_weight_update",
    "TrajectoryEncoder",
]

fused_weight_encode: Callable | None = None
fused_graph_message_pass: Callable | None = None
fused_weight_update: Callable | None = None
TrajectoryEncoder: type | None = None

if not TYPE_CHECKING and triton_supported():
    try:
        import triton
        import triton.language as tl
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        # ============================================================
        # Trajectory Encoding Kernel
        # ============================================================
        @triton.jit
        def _trajectory_encode_kernel(
            # Input pointers
            W_ptr,              # (num_params,) current weights
            Delta_ptr,          # (history_len, num_params) weight deltas
            Grad_ptr,           # (history_len, num_params) gradients
            # Projection matrices (learned)
            Proj_W_ptr,         # (input_features, embed_dim) projection for weight features
            Proj_bias_ptr,      # (embed_dim,) projection bias
            # Output
            Out_ptr,            # (num_params, embed_dim) output embeddings
            # Dimensions
            num_params,
            history_len: tl.constexpr,
            embed_dim: tl.constexpr,
            input_features: tl.constexpr,  # Number of extracted features
            has_grads: tl.constexpr,
            # Block config
            BLOCK_SIZE: tl.constexpr,
        ):
            """Fused trajectory encoding with learned projections.

            Extracts rich features from weight trajectories and projects
            them through a learned linear layer.

            Features extracted per weight:
            - Current value (normalized)
            - Delta statistics: mean, std, min, max, trend (linear slope)
            - Gradient statistics: mean, std, sign consistency
            - Second-order: acceleration, jerk estimates
            """
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < num_params

            # Load current weight
            w = tl.load(W_ptr + offs, mask=mask, other=0.0)

            # ---- Compute delta statistics ----
            delta_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            delta_sq_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            delta_min = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
            delta_max = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

            # For trend estimation: sum of t*delta and sum of t
            weighted_delta_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            t_sum = 0.0
            t_sq_sum = 0.0

            # Previous delta for acceleration
            prev_delta = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            accel_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            # Previous acceleration for jerk
            prev_accel = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            jerk_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            for h in range(history_len):
                delta = tl.load(
                    Delta_ptr + h * num_params + offs,
                    mask=mask,
                    other=0.0,
                )

                delta_sum += delta
                delta_sq_sum += delta * delta
                delta_min = tl.minimum(delta_min, delta)
                delta_max = tl.maximum(delta_max, delta)

                # Weighted sum for trend (linear regression)
                t = float(h)
                weighted_delta_sum += t * delta
                t_sum += t
                t_sq_sum += t * t

                # Acceleration (second derivative)
                if h > 0:
                    accel = delta - prev_delta
                    accel_sum += accel * accel

                    # Jerk (third derivative)
                    if h > 1:
                        jerk = accel - prev_accel
                        jerk_sum += jerk * jerk
                    prev_accel = accel

                prev_delta = delta

            # Compute statistics
            n = float(history_len)
            delta_mean = delta_sum / n
            delta_var = delta_sq_sum / n - delta_mean * delta_mean
            delta_std = tl.sqrt(tl.maximum(delta_var, 1e-8))
            delta_range = delta_max - delta_min

            # Trend (slope of linear fit): beta = (sum(t*y) - n*mean_t*mean_y) / (sum(t^2) - n*mean_t^2)
            mean_t = t_sum / n
            denom = t_sq_sum - n * mean_t * mean_t
            trend = tl.where(
                tl.abs(denom) > 1e-8,
                (weighted_delta_sum - n * mean_t * delta_mean) / denom,
                tl.zeros((BLOCK_SIZE,), dtype=tl.float32),
            )

            # Velocity (cumulative delta)
            velocity = delta_sum

            # Acceleration magnitude
            accel_mag = tl.sqrt(accel_sum / tl.maximum(n - 1, 1.0))

            # Jerk magnitude
            jerk_mag = tl.sqrt(jerk_sum / tl.maximum(n - 2, 1.0))

            # ---- Compute gradient statistics if available ----
            if has_grads:
                grad_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                grad_sq_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                grad_sign_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

                for h in range(history_len):
                    grad = tl.load(
                        Grad_ptr + h * num_params + offs,
                        mask=mask,
                        other=0.0,
                    )
                    grad_sum += grad
                    grad_sq_sum += grad * grad
                    grad_sign_sum += tl.where(grad > 0, 1.0, tl.where(grad < 0, -1.0, 0.0))

                grad_mean = grad_sum / n
                grad_var = grad_sq_sum / n - grad_mean * grad_mean
                grad_std = tl.sqrt(tl.maximum(grad_var, 1e-8))
                grad_sign_consistency = tl.abs(grad_sign_sum) / n  # 1.0 = all same sign
            else:
                grad_mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                grad_std = tl.ones((BLOCK_SIZE,), dtype=tl.float32)
                grad_sign_consistency = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            # ---- Normalize weight by its magnitude ----
            w_norm = w / (tl.abs(w) + 1e-8)
            w_mag = tl.log1p(tl.abs(w))

            # ---- Build feature vector and project ----
            # Features: [w_norm, w_mag, delta_mean, delta_std, delta_range,
            #            trend, velocity, accel_mag, jerk_mag,
            #            grad_mean, grad_std, grad_sign_consistency]
            # Total: 12 features

            # Store features temporarily
            features = tl.zeros((BLOCK_SIZE, input_features), dtype=tl.float32)

            # Project features through learned matrix
            for d in range(embed_dim):
                # Dot product of features with projection column
                proj_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

                # Feature 0: normalized weight
                proj_w = tl.load(Proj_W_ptr + 0 * embed_dim + d)
                proj_val += w_norm * proj_w

                # Feature 1: weight magnitude
                proj_w = tl.load(Proj_W_ptr + 1 * embed_dim + d)
                proj_val += w_mag * proj_w

                # Feature 2: delta mean
                proj_w = tl.load(Proj_W_ptr + 2 * embed_dim + d)
                proj_val += delta_mean * proj_w

                # Feature 3: delta std
                proj_w = tl.load(Proj_W_ptr + 3 * embed_dim + d)
                proj_val += delta_std * proj_w

                # Feature 4: delta range
                proj_w = tl.load(Proj_W_ptr + 4 * embed_dim + d)
                proj_val += delta_range * proj_w

                # Feature 5: trend
                proj_w = tl.load(Proj_W_ptr + 5 * embed_dim + d)
                proj_val += trend * proj_w

                # Feature 6: velocity
                proj_w = tl.load(Proj_W_ptr + 6 * embed_dim + d)
                proj_val += velocity * proj_w

                # Feature 7: acceleration magnitude
                proj_w = tl.load(Proj_W_ptr + 7 * embed_dim + d)
                proj_val += accel_mag * proj_w

                # Feature 8: jerk magnitude
                proj_w = tl.load(Proj_W_ptr + 8 * embed_dim + d)
                proj_val += jerk_mag * proj_w

                # Feature 9: gradient mean
                proj_w = tl.load(Proj_W_ptr + 9 * embed_dim + d)
                proj_val += grad_mean * proj_w

                # Feature 10: gradient std
                proj_w = tl.load(Proj_W_ptr + 10 * embed_dim + d)
                proj_val += grad_std * proj_w

                # Feature 11: gradient sign consistency
                proj_w = tl.load(Proj_W_ptr + 11 * embed_dim + d)
                proj_val += grad_sign_consistency * proj_w

                # Add bias
                bias = tl.load(Proj_bias_ptr + d)
                proj_val += bias

                # Apply GELU activation
                # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                sqrt_2_over_pi = 0.7978845608
                x = proj_val
                proj_val = 0.5 * x * (1.0 + tl.libdevice.tanh(
                    sqrt_2_over_pi * (x + 0.044715 * x * x * x)
                ))

                tl.store(
                    Out_ptr + offs * embed_dim + d,
                    proj_val,
                    mask=mask,
                )

        # ============================================================
        # Attention-Weighted Graph Message Passing Kernel
        # ============================================================
        @triton.jit
        def _attention_message_pass_kernel(
            # Node features
            Node_ptr,           # (num_nodes, node_dim)
            # Edge structure (CSR format for efficiency)
            RowPtr_ptr,         # (num_nodes + 1,) row pointers
            ColIdx_ptr,         # (num_edges,) column indices
            # Attention weights
            Attn_Q_ptr,         # (node_dim, head_dim) query projection
            Attn_K_ptr,         # (node_dim, head_dim) key projection
            Attn_V_ptr,         # (node_dim, node_dim) value projection
            # Output
            Out_ptr,            # (num_nodes, node_dim)
            # Dimensions
            num_nodes,
            node_dim: tl.constexpr,
            head_dim: tl.constexpr,
            scale,
            # Block config
            BLOCK_EDGES: tl.constexpr,
            MAX_DEGREE: tl.constexpr,
        ):
            """Attention-weighted message passing on weight graph.

            For each node, computes attention over its neighbors and
            aggregates their features weighted by attention scores.
            """
            # This kernel is implemented with *static* loop bounds (Triton requirement) by
            # bounding the maximum degree. We run a single-pass numerically-stable softmax
            # (online log-sum-exp) over neighbors for performance and to avoid a second pass.
            #
            # Performance note: we aggregate neighbor features first and apply the value
            # projection once:  sum_i a_i * (x_i @ V) == (sum_i a_i * x_i) @ V
            node_idx = tl.program_id(0)
            in_bounds = node_idx < num_nodes

            # Load row pointers for this node's edges (CSR).
            row_start = tl.load(RowPtr_ptr + node_idx, mask=in_bounds, other=0).to(tl.int32)
            row_end = tl.load(RowPtr_ptr + node_idx + 1, mask=in_bounds, other=0).to(tl.int32)
            num_neighbors = row_end - row_start

            offs_d = tl.arange(0, node_dim)
            offs_h = tl.arange(0, head_dim)

            # Load this node's features.
            node_feat = tl.load(
                Node_ptr + node_idx * node_dim + offs_d,
                mask=in_bounds,
                other=0.0,
            ).to(tl.float32)

            # Compute query for this node: q = x @ Q
            query = tl.zeros((head_dim,), dtype=tl.float32)
            for d in tl.static_range(0, node_dim):
                q_proj = tl.load(
                    Attn_Q_ptr + d * head_dim + offs_h,
                    mask=offs_h < head_dim,
                    other=0.0,
                ).to(tl.float32)
                query += node_feat[d] * q_proj

            # Online softmax accumulators (scalar m, scalar l, vector z).
            m = tl.full((), float("-inf"), dtype=tl.float32)
            l = tl.zeros((), dtype=tl.float32)
            z = tl.zeros((node_dim,), dtype=tl.float32)

            # Process neighbors in fixed blocks up to MAX_DEGREE.
            for e_base in tl.static_range(0, MAX_DEGREE, BLOCK_EDGES):
                e_offs = e_base + tl.arange(0, BLOCK_EDGES)
                e_mask = in_bounds & (e_offs < num_neighbors)

                neighbor_idx = tl.load(
                    ColIdx_ptr + row_start + e_offs,
                    mask=e_mask,
                    other=0,
                ).to(tl.int32)

                # Load neighbor features for this edge block: (BLOCK_EDGES, node_dim)
                n_ptrs = Node_ptr + neighbor_idx[:, None] * node_dim + offs_d[None, :]
                n_feat = tl.load(n_ptrs, mask=e_mask[:, None], other=0.0).to(tl.float32)

                # Compute keys for this block: k = x @ K  -> (BLOCK_EDGES, head_dim)
                key = tl.zeros((BLOCK_EDGES, head_dim), dtype=tl.float32)
                for d in tl.static_range(0, node_dim):
                    k_proj = tl.load(
                        Attn_K_ptr + d * head_dim + offs_h,
                        mask=offs_h < head_dim,
                        other=0.0,
                    ).to(tl.float32)
                    key += n_feat[:, d][:, None] * k_proj[None, :]

                # Scores for this block: (BLOCK_EDGES,)
                score = tl.sum(key * query[None, :], axis=1) * scale
                score = tl.where(e_mask, score, float("-inf"))

                # Online log-sum-exp update.
                m_new = tl.maximum(m, tl.max(score, axis=0))
                alpha = tl.exp(m - m_new)
                exp_scores = tl.exp(score - m_new)  # masked -inf -> 0

                l = l * alpha + tl.sum(exp_scores, axis=0)
                z = z * alpha + tl.sum(n_feat * exp_scores[:, None], axis=0)
                m = m_new

            # Normalize aggregated neighbor features.
            inv_l = tl.where(l > 0.0, 1.0 / l, 0.0)
            attn_feat = z * inv_l

            # Apply value projection once: out_feat = attn_feat @ V
            agg = tl.zeros((node_dim,), dtype=tl.float32)
            for d_in in tl.static_range(0, node_dim):
                v_row = tl.load(
                    Attn_V_ptr + d_in * node_dim + offs_d,
                    mask=offs_d < node_dim,
                    other=0.0,
                ).to(tl.float32)
                agg += attn_feat[d_in] * v_row

            out = node_feat + agg
            tl.store(
                Out_ptr + node_idx * node_dim + offs_d,
                out,
                mask=in_bounds,
            )

        # ============================================================
        # Fused Weight Update Kernel
        # ============================================================
        @triton.jit
        def _fused_weight_update_kernel(
            # Current weights
            W_ptr,              # (num_params,)
            # Predicted deltas
            Delta_ptr,          # (num_params,)
            # Momentum buffer
            Mom_ptr,            # (num_params,)
            # Output
            Out_ptr,            # (num_params,)
            # Params
            num_params,
            horizon,
            momentum: tl.constexpr,
            clip_value: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            """Fused weight update with momentum and gradient clipping."""
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < num_params

            # Load values
            w = tl.load(W_ptr + offs, mask=mask, other=0.0)
            delta = tl.load(Delta_ptr + offs, mask=mask, other=0.0)
            mom = tl.load(Mom_ptr + offs, mask=mask, other=0.0)

            # Clip delta
            delta = tl.maximum(tl.minimum(delta, clip_value), -clip_value)

            # Update momentum
            new_mom = momentum * mom + (1.0 - momentum) * delta

            # Scale by horizon and apply
            update = new_mom * horizon
            new_w = w + update

            # Store results
            tl.store(Out_ptr + offs, new_w, mask=mask)
            tl.store(Mom_ptr + offs, new_mom, mask=mask)

        # ============================================================
        # Python Wrappers
        # ============================================================

        class TrajectoryEncoder(nn.Module):
            """Learned trajectory encoder backed by Triton kernel."""

            INPUT_FEATURES = 12  # Number of features extracted from trajectory

            def __init__(self, embed_dim: int = 64):
                super().__init__()
                self.embed_dim = embed_dim

                # Learned projection
                self.proj_weight = nn.Parameter(
                    torch.randn(self.INPUT_FEATURES, embed_dim) * 0.02
                )
                self.proj_bias = nn.Parameter(torch.zeros(embed_dim))

            def forward(
                self,
                weights: torch.Tensor,
                deltas: torch.Tensor,
                grads: torch.Tensor | None = None,
            ) -> torch.Tensor:
                """Encode trajectories using fused kernel.

                Args:
                    weights: Current weights (num_params,)
                    deltas: Weight deltas (history_len, num_params)
                    grads: Gradients (history_len, num_params) or None

                Returns:
                    Embeddings (num_params, embed_dim)
                """
                num_params = weights.numel()
                history_len = deltas.shape[0]

                out = torch.empty(
                    num_params, self.embed_dim,
                    device=weights.device, dtype=torch.float32,
                )

                BLOCK_SIZE = 256
                grid = ((num_params + BLOCK_SIZE - 1) // BLOCK_SIZE,)

                # Handle grads
                if grads is None:
                    grads = torch.zeros_like(deltas)
                    has_grads = False
                else:
                    has_grads = True

                _trajectory_encode_kernel[grid](
                    weights.flatten().contiguous(),
                    deltas.view(history_len, -1).contiguous(),
                    grads.view(history_len, -1).contiguous(),
                    self.proj_weight.contiguous(),
                    self.proj_bias.contiguous(),
                    out,
                    num_params,
                    history_len,
                    self.embed_dim,
                    self.INPUT_FEATURES,
                    has_grads,
                    BLOCK_SIZE,
                )

                return out

        def fused_weight_encode(weights, deltas, grads=None, proj_weight=None, proj_bias=None, embed_dim=64):
            """Encode weight trajectories using fused Triton kernel.

            Args:
                weights: Current weights (num_params,)
                deltas: Weight deltas (history_len, num_params)
                grads: Optional gradients (history_len, num_params)
                proj_weight: Learned projection (input_features, embed_dim)
                proj_bias: Learned bias (embed_dim,)
                embed_dim: Output embedding dimension

            Returns:
                Embeddings (num_params, embed_dim)
            """
            num_params = weights.numel()
            history_len = deltas.shape[0]
            input_features = 12

            # Use default projection if not provided
            if proj_weight is None:
                proj_weight = torch.randn(
                    input_features, embed_dim,
                    device=weights.device, dtype=torch.float32,
                ) * 0.02
            if proj_bias is None:
                proj_bias = torch.zeros(embed_dim, device=weights.device, dtype=torch.float32)

            out = torch.empty(num_params, embed_dim, device=weights.device, dtype=torch.float32)

            BLOCK_SIZE = 256
            grid = ((num_params + BLOCK_SIZE - 1) // BLOCK_SIZE,)

            if grads is None:
                grads = torch.zeros_like(deltas)
                has_grads = False
            else:
                has_grads = True

            _trajectory_encode_kernel[grid](
                weights.flatten().contiguous(),
                deltas.view(history_len, -1).contiguous(),
                grads.view(history_len, -1).contiguous(),
                proj_weight.contiguous(),
                proj_bias.contiguous(),
                out,
                num_params,
                history_len,
                embed_dim,
                input_features,
                has_grads,
                BLOCK_SIZE,
            )

            return out

        def fused_graph_message_pass(
            node_features: torch.Tensor,
            row_ptr: torch.Tensor,
            col_idx: torch.Tensor,
            attn_q: torch.Tensor,
            attn_k: torch.Tensor,
            attn_v: torch.Tensor,
            *,
            max_degree: int = 256,
        ) -> torch.Tensor:
            """Attention-weighted graph message passing.

            Args:
                node_features: (num_nodes, node_dim)
                row_ptr: CSR row pointers (num_nodes + 1,)
                col_idx: CSR column indices (num_edges,)
                attn_q: Query projection (node_dim, head_dim)
                attn_k: Key projection (node_dim, head_dim)
                attn_v: Value projection (node_dim, node_dim)

            Returns:
                Updated node features (num_nodes, node_dim)
            """
            num_nodes, node_dim = node_features.shape
            head_dim = attn_q.shape[1]
            scale = 1.0 / (head_dim ** 0.5)

            out = torch.empty_like(node_features)

            grid = (num_nodes,)
            BLOCK_EDGES = 32
            max_degree_i = int(max_degree)
            # Ensure MAX_DEGREE is a multiple of BLOCK_EDGES for tl.static_range.
            MAX_DEGREE = int(math.ceil(max_degree_i / BLOCK_EDGES) * BLOCK_EDGES)

            _attention_message_pass_kernel[grid](
                node_features.contiguous(),
                row_ptr.contiguous(),
                col_idx.contiguous(),
                attn_q.contiguous(),
                attn_k.contiguous(),
                attn_v.contiguous(),
                out,
                num_nodes,
                node_dim,
                head_dim,
                scale,
                BLOCK_EDGES,
                MAX_DEGREE,
                num_warps=4,
            )

            return out

        def fused_weight_update(
            weights: torch.Tensor,
            predicted_delta: torch.Tensor,
            momentum_buffer: torch.Tensor,
            horizon: int,
            momentum: float = 0.9,
            clip_value: float = 1.0,
        ) -> torch.Tensor:
            """Apply predicted weight update with momentum.

            Args:
                weights: Current weights (num_params,)
                predicted_delta: Predicted delta per step (num_params,)
                momentum_buffer: Momentum buffer (num_params,) - modified in place
                horizon: Number of steps to skip
                momentum: Momentum coefficient
                clip_value: Gradient clipping value

            Returns:
                Updated weights (num_params,)
            """
            if not momentum_buffer.is_contiguous():
                raise RuntimeError(
                    "fused_weight_update: momentum_buffer must be contiguous for in-place updates "
                    "(got non-contiguous tensor; refusing to silently update a copy)."
                )
            num_params = weights.numel()
            out = torch.empty_like(weights)

            BLOCK_SIZE = 1024
            grid = ((num_params + BLOCK_SIZE - 1) // BLOCK_SIZE,)

            _fused_weight_update_kernel[grid](
                weights.flatten().contiguous(),
                predicted_delta.flatten().contiguous(),
                momentum_buffer.view(-1),
                out,
                num_params,
                horizon,
                momentum,
                clip_value,
                BLOCK_SIZE,
            )

            return out.view(weights.shape)
