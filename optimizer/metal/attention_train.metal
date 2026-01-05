#include <metal_stdlib>
using namespace metal;

// Fused attention training (Metal/MPS).
//
// This implements full-sequence scaled dot-product attention for training:
//   out[b,h,q,:] = sum_k softmax(scale * dot(q[b,h,q,:], k[b,h,k,:]))[k] * dropout_mask[q,k] * v[b,h,k,:]
//
// Key properties:
// - Numerically stable: online blockwise log-sum-exp softmax.
// - No score materialization: weights are staged per token block in threadgroup memory.
// - Deterministic dropout: regenerated from a saved seed and the (q,k) pair indices.

struct AttnParams {
    uint B;
    uint H;
    uint T;
    uint D;
    float scale;
    uint causal;
    float dropout_p;
    uint seed;
    // Strides in ELEMENTS for (B,H,T,D) tensors (last dim contiguous).
    uint q_stride_b;
    uint q_stride_h;
    uint q_stride_t;
    uint k_stride_b;
    uint k_stride_h;
    uint k_stride_t;
    uint v_stride_b;
    uint v_stride_h;
    uint v_stride_t;
    uint o_stride_b;
    uint o_stride_h;
    uint o_stride_t;
    uint lse_stride_b;
    uint lse_stride_h;
    uint lse_stride_t;
};

constexpr uint TG = 256;

inline float dot_half(device const half* a, device const half* b, uint dim) {
    // Vectorize within a thread using packed_half2 to leverage SIMD ops.
    float2 acc2 = float2(0.0f);
    const uint n2 = dim / 2;
    device const packed_half2* ap2 = reinterpret_cast<device const packed_half2*>(a);
    device const packed_half2* bp2 = reinterpret_cast<device const packed_half2*>(b);
    for (uint i = 0; i < n2; ++i) {
        const half2 ha = half2(ap2[i]);
        const half2 hb = half2(bp2[i]);
        acc2 += float2(ha) * float2(hb);
    }
    float acc = acc2.x + acc2.y;
    if (dim & 1u) {
        acc += float(a[dim - 1]) * float(b[dim - 1]);
    }
    return acc;
}

inline uint splitmix32(uint x) {
    uint z = x + 0x9E3779B9u;
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

inline float rand01(uint seed, uint row, uint q, uint k, uint T) {
    // A deterministic, stateless RNG keyed by (seed,row,q,k).
    // We avoid floating-point state to keep exact reproducibility.
    const uint pair = q * T + k;
    const uint x = seed ^ (row * 0xA511E9B3u) ^ splitmix32(pair);
    const uint r = splitmix32(x);
    // 24-bit mantissa uniform: [0,1)
    return float(r & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

kernel void attn_train_fwd_fp16(
    device const half* q [[ buffer(0) ]],
    device const half* k [[ buffer(1) ]],
    device const half* v [[ buffer(2) ]],
    device half* out     [[ buffer(3) ]],
    device float* lse    [[ buffer(4) ]],
    constant AttnParams& p [[ buffer(5) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]],
    uint sg [[ simdgroup_index_in_threadgroup ]],
    uint sgs [[ simdgroups_per_threadgroup ]],
    uint3 tgid [[ threadgroup_position_in_grid ]]
) {
    // Grid: (T, H, B) -> (q_idx, head, batch)
    const uint q_idx = tgid.x;
    const uint head = tgid.y;
    const uint batch = tgid.z;
    const uint row = batch * p.H + head;

    device const half* q_vec = q + batch * p.q_stride_b + head * p.q_stride_h + q_idx * p.q_stride_t;
    const bool compute_out = tid < p.D;
    float out_acc = 0.0f;

    threadgroup float tg_max[TG];
    threadgroup float tg_sum[TG];
    threadgroup float weights_raw[TG];
    threadgroup float weights_drop[TG];
    threadgroup float shared_m;
    threadgroup float shared_d;
    threadgroup float shared_alpha;
    threadgroup float shared_beta;
    threadgroup float shared_block_m;

    const bool lane0 = (lane == 0);
    if (tid == 0) {
        shared_m = -INFINITY;
        shared_d = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool use_dropout = p.dropout_p > 0.0f;
    const float keep_prob = 1.0f - p.dropout_p;
    const float inv_keep = use_dropout ? (1.0f / keep_prob) : 1.0f;

    for (uint block = 0; block < p.T; block += TG) {
        const uint k_idx = block + tid;
        float s = -INFINITY;
        if (k_idx < p.T) {
            if (!(p.causal && (k_idx > q_idx))) {
                device const half* k_vec = k + batch * p.k_stride_b + head * p.k_stride_h + k_idx * p.k_stride_t;
                s = dot_half(q_vec, k_vec, p.D) * p.scale;
            }
        }

        float sg_m = simd_max(s);
        if (lane0) {
            tg_max[sg] = sg_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float m_blk = -INFINITY;
            for (uint i = 0; i < sgs; ++i) {
                m_blk = max(m_blk, tg_max[i]);
            }
            shared_block_m = m_blk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float m_blk2 = shared_block_m;
        const float w_raw = (k_idx < p.T) ? exp(s - m_blk2) : 0.0f;
        weights_raw[tid] = w_raw;

        float w_drop = w_raw;
        if (use_dropout && (k_idx < p.T) && !(p.causal && (k_idx > q_idx))) {
            const float r = rand01(p.seed, row, q_idx, k_idx, p.T);
            const bool keep = r < keep_prob;
            w_drop = keep ? (w_raw * inv_keep) : 0.0f;
        }
        weights_drop[tid] = w_drop;

        float sg_d = simd_sum(w_raw);
        if (lane0) {
            tg_sum[sg] = sg_d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float d_blk = 0.0f;
            for (uint i = 0; i < sgs; ++i) {
                d_blk += tg_sum[i];
            }
            const float m_prev = shared_m;
            const float d_prev = shared_d;
            const float m_new = max(m_prev, m_blk2);
            const float alpha = exp(m_prev - m_new); // exp(-inf) -> 0
            const float beta = exp(m_blk2 - m_new);
            shared_m = m_new;
            shared_d = d_prev * alpha + d_blk * beta;
            shared_alpha = alpha;
            shared_beta = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = shared_alpha;
        const float beta = shared_beta;

        if (compute_out) {
            float acc_blk = 0.0f;
            const uint valid = min(TG, p.T - block);
            device const half* v_block = v + batch * p.v_stride_b + head * p.v_stride_h + block * p.v_stride_t;
            for (uint i = 0; i < valid; ++i) {
                const float wi = weights_drop[i];
                const half vi = v_block[i * p.v_stride_t + tid];
                acc_blk += wi * float(vi);
            }
            out_acc = out_acc * alpha + acc_blk * beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = shared_d;

    if (compute_out) {
        const float y = (denom > 0.0f) ? (out_acc / denom) : 0.0f;
        device half* o_vec = out + batch * p.o_stride_b + head * p.o_stride_h + q_idx * p.o_stride_t;
        o_vec[tid] = half(y);
    }
    if (tid == 0) {
        device float* lse_vec = lse + batch * p.lse_stride_b + head * p.lse_stride_h + q_idx * p.lse_stride_t;
        lse_vec[0] = (denom > 0.0f) ? (log(denom) + shared_m) : -INFINITY;
    }
}

kernel void attn_train_bwd_preprocess_fp16(
    device const half* out     [[ buffer(0) ]],
    device const half* grad_out [[ buffer(1) ]],
    device float* delta        [[ buffer(2) ]],
    constant AttnParams& p     [[ buffer(3) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]],
    uint sg [[ simdgroup_index_in_threadgroup ]],
    uint sgs [[ simdgroups_per_threadgroup ]],
    uint3 tgid [[ threadgroup_position_in_grid ]]
) {
    const uint q_idx = tgid.x;
    const uint head = tgid.y;
    const uint batch = tgid.z;

    threadgroup float tg_sum[TG];
    const bool lane0 = (lane == 0);

    float x = 0.0f;
    if (tid < p.D) {
        device const half* o_vec = out + batch * p.o_stride_b + head * p.o_stride_h + q_idx * p.o_stride_t;
        device const half* go_vec = grad_out + batch * p.o_stride_b + head * p.o_stride_h + q_idx * p.o_stride_t;
        x = float(o_vec[tid]) * float(go_vec[tid]);
    }
    float sg_sum = simd_sum(x);
    if (lane0) {
        tg_sum[sg] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < sgs; ++i) {
            s += tg_sum[i];
        }
        device float* d_vec = delta + batch * p.lse_stride_b + head * p.lse_stride_h + q_idx * p.lse_stride_t;
        d_vec[0] = s;
    }
}

kernel void attn_train_bwd_dkv_fp16(
    device const half* q [[ buffer(0) ]],
    device const half* k [[ buffer(1) ]],
    device const half* v [[ buffer(2) ]],
    device const half* grad_out [[ buffer(3) ]],
    device const float* lse [[ buffer(4) ]],
    device const float* delta [[ buffer(5) ]],
    device half* grad_k [[ buffer(6) ]],
    device half* grad_v [[ buffer(7) ]],
    constant AttnParams& p [[ buffer(8) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]],
    uint sg [[ simdgroup_index_in_threadgroup ]],
    uint sgs [[ simdgroups_per_threadgroup ]],
    uint3 tgid [[ threadgroup_position_in_grid ]]
) {
    // Grid: (T, H, B) -> (k_idx, head, batch)
    const uint k_idx = tgid.x;
    const uint head = tgid.y;
    const uint batch = tgid.z;
    const uint row = batch * p.H + head;

    if (k_idx >= p.T) {
        return;
    }
    const bool active_d = tid < p.D;

    device const half* k_vec = k + batch * p.k_stride_b + head * p.k_stride_h + k_idx * p.k_stride_t;
    device const half* v_vec = v + batch * p.v_stride_b + head * p.v_stride_h + k_idx * p.v_stride_t;

    float dk = 0.0f;
    float dv = 0.0f;

    threadgroup float tg_sum[TG];
    threadgroup float shared_s;
    threadgroup float shared_dp;
    const bool lane0 = (lane == 0);

    const bool use_dropout = p.dropout_p > 0.0f;
    const float keep_prob = 1.0f - p.dropout_p;
    const float inv_keep = use_dropout ? (1.0f / keep_prob) : 1.0f;

    for (uint q_idx = 0; q_idx < p.T; ++q_idx) {
        if (p.causal && (k_idx > q_idx)) {
            continue;
        }

        device const half* q_vec = q + batch * p.q_stride_b + head * p.q_stride_h + q_idx * p.q_stride_t;
        device const half* go_vec = grad_out + batch * p.o_stride_b + head * p.o_stride_h + q_idx * p.o_stride_t;

        device const float* lse_vec = lse + batch * p.lse_stride_b + head * p.lse_stride_h + q_idx * p.lse_stride_t;
        device const float* del_vec = delta + batch * p.lse_stride_b + head * p.lse_stride_h + q_idx * p.lse_stride_t;
        const float lse_q = lse_vec[0];
        const float delta_q = del_vec[0];

        // Compute s = dot(q_vec, k_vec) * scale once per (q,k) using a reduction across `tid`.
        const float s_contrib = active_d ? (float(q_vec[tid]) * float(k_vec[tid])) : 0.0f;
        float sg_s = simd_sum(s_contrib);
        if (lane0) {
            tg_sum[sg] = sg_s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s_total = 0.0f;
            for (uint i = 0; i < sgs; ++i) {
                s_total += tg_sum[i];
            }
            shared_s = s_total * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float s = shared_s;
        const float p_raw = exp(s - lse_q);
        float w = p_raw;
        if (use_dropout) {
            const float r = rand01(p.seed, row, q_idx, k_idx, p.T);
            const bool keep = r < keep_prob;
            w = keep ? (p_raw * inv_keep) : 0.0f;
        }

        // dp = sum_d(grad_out[q,d] * v[k,d])
        const float contrib = active_d ? (float(go_vec[tid]) * float(v_vec[tid])) : 0.0f;
        float sgs = simd_sum(contrib);
        if (lane0) {
            tg_sum[sg] = sgs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float dp = 0.0f;
            for (uint i = 0; i < sgs; ++i) {
                dp += tg_sum[i];
            }
            shared_dp = dp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float dp = shared_dp;
        const float ds = (w * (dp - delta_q)) * p.scale;

        if (active_d) {
            dv += w * float(go_vec[tid]);
            dk += ds * float(q_vec[tid]);
        }
    }

    device half* gk_vec = grad_k + batch * p.k_stride_b + head * p.k_stride_h + k_idx * p.k_stride_t;
    device half* gv_vec = grad_v + batch * p.v_stride_b + head * p.v_stride_h + k_idx * p.v_stride_t;
    if (active_d) {
        gk_vec[tid] = half(dk);
        gv_vec[tid] = half(dv);
    }
}

kernel void attn_train_bwd_dq_fp16(
    device const half* q [[ buffer(0) ]],
    device const half* k [[ buffer(1) ]],
    device const half* v [[ buffer(2) ]],
    device const half* grad_out [[ buffer(3) ]],
    device const float* lse [[ buffer(4) ]],
    device const float* delta [[ buffer(5) ]],
    device half* grad_q [[ buffer(6) ]],
    constant AttnParams& p [[ buffer(7) ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint lane [[ thread_index_in_simdgroup ]],
    uint sg [[ simdgroup_index_in_threadgroup ]],
    uint sgs [[ simdgroups_per_threadgroup ]],
    uint3 tgid [[ threadgroup_position_in_grid ]]
) {
    // Grid: (T, H, B) -> (q_idx, head, batch)
    const uint q_idx = tgid.x;
    const uint head = tgid.y;
    const uint batch = tgid.z;
    const uint row = batch * p.H + head;

    if (q_idx >= p.T) {
        return;
    }
    const bool active_d = tid < p.D;

    device const half* q_vec = q + batch * p.q_stride_b + head * p.q_stride_h + q_idx * p.q_stride_t;
    device const half* go_vec = grad_out + batch * p.o_stride_b + head * p.o_stride_h + q_idx * p.o_stride_t;
    device const float* lse_vec = lse + batch * p.lse_stride_b + head * p.lse_stride_h + q_idx * p.lse_stride_t;
    device const float* del_vec = delta + batch * p.lse_stride_b + head * p.lse_stride_h + q_idx * p.lse_stride_t;
    const float lse_q = lse_vec[0];
    const float delta_q = del_vec[0];

    float dq = 0.0f;

    threadgroup float tg_sum[TG];
    threadgroup float shared_s;
    threadgroup float shared_dp;
    const bool lane0 = (lane == 0);

    const bool use_dropout = p.dropout_p > 0.0f;
    const float keep_prob = 1.0f - p.dropout_p;
    const float inv_keep = use_dropout ? (1.0f / keep_prob) : 1.0f;

    for (uint k_idx = 0; k_idx < p.T; ++k_idx) {
        if (p.causal && (k_idx > q_idx)) {
            break;
        }
        device const half* k_vec = k + batch * p.k_stride_b + head * p.k_stride_h + k_idx * p.k_stride_t;
        device const half* v_vec = v + batch * p.v_stride_b + head * p.v_stride_h + k_idx * p.v_stride_t;

        // Compute s = dot(q_vec, k_vec) * scale once per (q,k) using a reduction across `tid`.
        const float s_contrib = active_d ? (float(q_vec[tid]) * float(k_vec[tid])) : 0.0f;
        float sg_s = simd_sum(s_contrib);
        if (lane0) {
            tg_sum[sg] = sg_s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s_total = 0.0f;
            for (uint i = 0; i < sgs; ++i) {
                s_total += tg_sum[i];
            }
            shared_s = s_total * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float s = shared_s;
        const float p_raw = exp(s - lse_q);
        float w = p_raw;
        if (use_dropout) {
            const float r = rand01(p.seed, row, q_idx, k_idx, p.T);
            const bool keep = r < keep_prob;
            w = keep ? (p_raw * inv_keep) : 0.0f;
        }

        const float contrib = active_d ? (float(go_vec[tid]) * float(v_vec[tid])) : 0.0f;
        float sgs = simd_sum(contrib);
        if (lane0) {
            tg_sum[sg] = sgs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float dp = 0.0f;
            for (uint i = 0; i < sgs; ++i) {
                dp += tg_sum[i];
            }
            shared_dp = dp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float dp = shared_dp;
        const float ds = (w * (dp - delta_q)) * p.scale;
        if (active_d) {
            dq += ds * float(k_vec[tid]);
        }
    }

    device half* gq_vec = grad_q + batch * p.q_stride_b + head * p.q_stride_h + q_idx * p.q_stride_t;
    if (active_d) {
        gq_vec[tid] = half(dq);
    }
}
