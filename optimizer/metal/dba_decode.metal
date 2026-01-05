#include <metal_stdlib>
using namespace metal;

// Parameters for fused DBA decode.
struct DBAParams {
    uint sem_head_dim;
    uint geo_head_dim;
    uint v_head_dim;
    uint n_heads;
    uint seq_len;
    float sem_scale;
    float geo_scale;
    // Strides for (B,S,D) tensors in units of elements (not bytes).
    // We assume the last dimension is contiguous (stride == 1).
    uint ksem_stride_b;
    uint ksem_stride_t;
    uint kgeo_stride_b;
    uint kgeo_stride_t;
    uint v_stride_b;
    uint v_stride_t;
};

inline float dot_half(device const half* a, device const half* b, uint dim) {
    // Vectorize within a thread using packed_half2 to leverage SIMD ops.
    //
    // IMPORTANT: packed_half2 loads require 4-byte alignment. The incoming pointers
    // can be derived from head/offset arithmetic (e.g. head * sem_hd) and may be
    // only 2-byte aligned. Only vectorize when both pointers are 4-byte aligned
    // AND dim is even; otherwise fall back to a scalar loop.
    const bool dim_even = ((dim & 1u) == 0u);
    const bool a_aligned4 = ((reinterpret_cast<ulong>(a) & 0x3ul) == 0ul);
    const bool b_aligned4 = ((reinterpret_cast<ulong>(b) & 0x3ul) == 0ul);
    const bool can_vec = dim_even && a_aligned4 && b_aligned4;

    device const packed_half2* ap2 = nullptr;
    device const packed_half2* bp2 = nullptr;

    if (!can_vec) {
        float acc = 0.0f;
        for (uint i = 0; i < dim; ++i) {
            acc += float(a[i]) * float(b[i]);
        }
        return acc;
    }

    float2 acc2 = float2(0.0f);
    const uint n2 = dim / 2;
    ap2 = reinterpret_cast<device const packed_half2*>(a);
    bp2 = reinterpret_cast<device const packed_half2*>(b);
    for (uint i = 0; i < n2; ++i) {
        const half2 ha = half2(ap2[i]);
        const half2 hb = half2(bp2[i]);
        acc2 += float2(ha) * float2(hb);
    }
    return acc2.x + acc2.y;
}

// Fused DBA decode for fp16 caches.
//
// Shapes (flattened contiguous):
//  - q_sem: (B,H,sem_hd)           -> q_sem[(b*H+h)*sem_hd + i]
//  - q_geo: (B,H,geo_hd)
//  - k_sem: (B,S,H*sem_hd)         -> k_sem[(b*S + t)*(H*sem_hd) + h*sem_hd + i]
//  - k_geo: (B,S,H*geo_hd)
//  - v:     (B,S,H*v_hd)
//  - out:   (B,H,v_hd)
//
// Numerics:
//  - Two-pass softmax (max, then exp-sum) for stability.
//  - No score materialization; weights are staged per token-block in threadgroup memory.
kernel void dba_decode_fp16(
    device const half* q_sem [[ buffer(0) ]],
    device const half* k_sem [[ buffer(1) ]],
    device const half* q_geo [[ buffer(2) ]],
    device const half* k_geo [[ buffer(3) ]],
    device const half* v     [[ buffer(4) ]],
    device half* out         [[ buffer(5) ]],
    constant DBAParams& p    [[ buffer(6) ]],
    // Use a scalar thread index for a 1D threadgroup.
    // (thread_position_in_threadgroup is uint3; mixing uint + uint3 in inputs is rejected
    // by newer Metal toolchains.)
    uint tid                [[ thread_index_in_threadgroup ]],
    uint lane               [[ thread_index_in_simdgroup ]],
    uint sg                 [[ simdgroup_index_in_threadgroup ]],
    uint sgs                [[ simdgroups_per_threadgroup ]],
    uint3 tgid              [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;

    // Grid: (1, n_heads, batch_size)
    const uint head = tgid.y;
    const uint batch = tgid.z;

    const uint H = p.n_heads;
    const uint sem_hd = p.sem_head_dim;
    const uint geo_hd = p.geo_head_dim;
    const uint v_hd = p.v_head_dim;
    const uint S = p.seq_len;

    const uint row = batch * H + head; // 0..(B*H-1)

    // NOTE: K/V tensors may be views into a larger preallocated cache buffer, so
    // their batch stride may exceed (S * token_stride). Use strides passed in.
    const uint ksem_stride_b = p.ksem_stride_b;
    const uint ksem_stride_tok = p.ksem_stride_t;
    const uint kgeo_stride_b = p.kgeo_stride_b;
    const uint kgeo_stride_tok = p.kgeo_stride_t;
    const uint v_stride_b = p.v_stride_b;
    const uint v_stride_tok = p.v_stride_t;

    device const half* qsem = q_sem + row * sem_hd;
    device const half* qgeo = q_geo + row * geo_hd;

    threadgroup float tg_max[TG];
    threadgroup float tg_sum[TG];
    threadgroup float weights[TG];
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

    // -------------------------------
    // Blockwise online softmax:
    // process the KV prefix in blocks of TG tokens and combine block stats:
    //   m_new = max(m, m_blk)
    //   d_new = d*exp(m-m_new) + d_blk*exp(m_blk-m_new)
    //   o_new = o*exp(m-m_new) + o_blk*exp(m_blk-m_new)
    // where (m_blk,d_blk,o_blk) are computed for the current token block.
    // -------------------------------
    float out_acc = 0.0f;
    const bool compute_out = tid < v_hd;

    for (uint block = 0; block < S; block += TG) {
        const uint t = block + tid;
        float s = -INFINITY;
        if (t < S) {
            device const half* ksem_t = k_sem + batch * ksem_stride_b + t * ksem_stride_tok + head * sem_hd;
            device const half* kgeo_t = k_geo + batch * kgeo_stride_b + t * kgeo_stride_tok + head * geo_hd;
            s = dot_half(qsem, ksem_t, sem_hd) * p.sem_scale + dot_half(qgeo, kgeo_t, geo_hd) * p.geo_scale;
        }
        // Block max over token scores in this block.
        float sg_max = simd_max(s);
        if (lane0) {
            tg_max[sg] = sg_max;
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
        const float w = (t < S) ? exp(s - m_blk2) : 0.0f;
        weights[tid] = w;

        // Block denom
        float sg_sum = simd_sum(w);
        if (lane0) {
            tg_sum[sg] = sg_sum;
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

        // Use weights[] to accumulate output per v-dimension.
        if (compute_out) {
            float acc_blk = 0.0f;
            const uint valid = min(TG, S - block);
            device const half* v_block = v + batch * v_stride_b + block * v_stride_tok + head * v_hd;
            for (uint i = 0; i < valid; ++i) {
                float wi = weights[i];
                half vi = v_block[i * v_stride_tok + tid];
                acc_blk += wi * float(vi);
            }
            out_acc = out_acc * alpha + acc_blk * beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = shared_d;

    if (compute_out) {
        // If denom is 0 (shouldn't happen for normal decode), write zeros.
        float y = (denom > 0.0f) ? (out_acc / denom) : 0.0f;
        out[row * v_hd + tid] = half(y);
    }
}

// Fused DBA decode for fp16 caches with an optional learned "null" KV token.
//
// This seeds the online softmax accumulator with a single extra KV entry
// (k_sem_null, k_geo_null, v_null) that is always available during decode.
kernel void dba_decode_fp16_null(
    device const half* q_sem [[ buffer(0) ]],
    device const half* k_sem [[ buffer(1) ]],
    device const half* q_geo [[ buffer(2) ]],
    device const half* k_geo [[ buffer(3) ]],
    device const half* v     [[ buffer(4) ]],
    device const half* k_sem_null [[ buffer(5) ]], // (B,H,sem_hd)
    device const half* k_geo_null [[ buffer(6) ]], // (B,H,geo_hd)
    device const half* v_null     [[ buffer(7) ]], // (B,H,v_hd)
    device half* out              [[ buffer(8) ]],
    constant DBAParams& p         [[ buffer(9) ]],
    uint tid                      [[ thread_index_in_threadgroup ]],
    uint lane                     [[ thread_index_in_simdgroup ]],
    uint sg                       [[ simdgroup_index_in_threadgroup ]],
    uint sgs                      [[ simdgroups_per_threadgroup ]],
    uint3 tgid                    [[ threadgroup_position_in_grid ]]
) {
    constexpr uint TG = 256;

    // Grid: (1, n_heads, batch_size)
    const uint head = tgid.y;
    const uint batch = tgid.z;

    const uint H = p.n_heads;
    const uint sem_hd = p.sem_head_dim;
    const uint geo_hd = p.geo_head_dim;
    const uint v_hd = p.v_head_dim;
    const uint S = p.seq_len;

    const uint row = batch * H + head; // 0..(B*H-1)

    const uint ksem_stride_b = p.ksem_stride_b;
    const uint ksem_stride_tok = p.ksem_stride_t;
    const uint kgeo_stride_b = p.kgeo_stride_b;
    const uint kgeo_stride_tok = p.kgeo_stride_t;
    const uint v_stride_b = p.v_stride_b;
    const uint v_stride_tok = p.v_stride_t;

    device const half* qsem = q_sem + row * sem_hd;
    device const half* qgeo = q_geo + row * geo_hd;

    // Null KV pointers are stored densely per (B,H).
    device const half* ksn = k_sem_null + row * sem_hd;
    device const half* kgn = k_geo_null + row * geo_hd;
    device const half* vn  = v_null + row * v_hd;

    threadgroup float tg_max[TG];
    threadgroup float tg_sum[TG];
    threadgroup float weights[TG];
    threadgroup float shared_m;
    threadgroup float shared_d;
    threadgroup float shared_alpha;
    threadgroup float shared_beta;
    threadgroup float shared_block_m;

    const bool lane0 = (lane == 0);

    // Seed online softmax with the null token.
    float out_acc = 0.0f;
    const bool compute_out = tid < v_hd;
    if (tid == 0) {
        const float s_null = dot_half(qsem, ksn, sem_hd) * p.sem_scale + dot_half(qgeo, kgn, geo_hd) * p.geo_scale;
        shared_m = s_null;
        shared_d = 1.0f; // exp(s_null - s_null)
    }
    if (compute_out) {
        out_acc = float(vn[tid]);
    }
    // `out_acc` is thread-local and does not depend on `shared_m/shared_d`.
    // This barrier ensures `shared_m/shared_d` (written by tid==0) are visible
    // to all threads before the subsequent loop reads them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block = 0; block < S; block += TG) {
        const uint t = block + tid;
        float s = -INFINITY;
        if (t < S) {
            device const half* ksem_t = k_sem + batch * ksem_stride_b + t * ksem_stride_tok + head * sem_hd;
            device const half* kgeo_t = k_geo + batch * kgeo_stride_b + t * kgeo_stride_tok + head * geo_hd;
            s = dot_half(qsem, ksem_t, sem_hd) * p.sem_scale + dot_half(qgeo, kgeo_t, geo_hd) * p.geo_scale;
        }

        float sg_max = simd_max(s);
        if (lane0) {
            tg_max[sg] = sg_max;
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
        const float w = (t < S) ? exp(s - m_blk2) : 0.0f;
        weights[tid] = w;

        float sg_sum = simd_sum(w);
        if (lane0) {
            tg_sum[sg] = sg_sum;
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
            const float alpha = exp(m_prev - m_new);
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
            const uint valid = min(TG, S - block);
            device const half* v_block = v + batch * v_stride_b + block * v_stride_tok + head * v_hd;
            for (uint i = 0; i < valid; ++i) {
                float wi = weights[i];
                half vi = v_block[i * v_stride_tok + tid];
                acc_blk += wi * float(vi);
            }
            out_acc = out_acc * alpha + acc_blk * beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = shared_d;

    if (compute_out) {
        float y = (denom > 0.0f) ? (out_acc / denom) : 0.0f;
        out[row * v_hd + tid] = half(y);
    }
}

