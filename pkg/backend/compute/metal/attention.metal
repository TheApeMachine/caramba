// Compile with:
// xcrun -sdk macosx metal -c attention.metal -o attention.air && xcrun -sdk macosx metallib attention.air -o attention.metallib

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// ---------------------------------------------------------------------------
// Scaled Dot-Product Attention (SDPA)
// One threadgroup per (batch, head) pair.
// thread t_idx handles query position t_idx.
// ---------------------------------------------------------------------------

kernel void sdpa_forward(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    constant int&       seq_len  [[buffer(4)]],
    constant int&       head_dim [[buffer(5)]],
    threadgroup float*  smem     [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= seq_len) return;

    int head_offset = (int)head_idx * seq_len * head_dim;
    float scale = rsqrt((float)head_dim);

    // Compute scores[t_idx, j] = dot(Q[t_idx], K[j]) * scale
    const device float* q_row = q + head_offset + (int)t_idx * head_dim;

    for (int j = 0; j < seq_len; j++) {
        const device float* k_row = k + head_offset + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        smem[(int)t_idx * seq_len + j] = dot * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax over row t_idx
    threadgroup float* row = smem + (int)t_idx * seq_len;
    float max_val = row[0];
    for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);

    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        row[j] = exp(row[j] - max_val);
        sum += row[j];
    }
    for (int j = 0; j < seq_len; j++) row[j] /= sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output[t_idx] = sum_j weights[j] * V[j]
    device float* out_row = out + head_offset + (int)t_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += row[j] * (v + head_offset + j * head_dim)[d];
        }
        out_row[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Multi-Query Attention (MQA)
// K and V have only 1 head; Q has num_heads heads.
// head_idx addresses Q heads; K/V use head 0.
// ---------------------------------------------------------------------------

kernel void mqa_forward(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    constant int&       seq_len  [[buffer(4)]],
    constant int&       head_dim [[buffer(5)]],
    threadgroup float*  smem     [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= seq_len) return;

    int q_head_offset  = (int)head_idx * seq_len * head_dim;
    int kv_head_offset = 0; // single KV head
    float scale = rsqrt((float)head_dim);

    const device float* q_row = q + q_head_offset + (int)t_idx * head_dim;

    for (int j = 0; j < seq_len; j++) {
        const device float* k_row = k + kv_head_offset + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
        smem[(int)t_idx * seq_len + j] = dot * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* row = smem + (int)t_idx * seq_len;
    float max_val = row[0];
    for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) { row[j] = exp(row[j] - max_val); sum += row[j]; }
    for (int j = 0; j < seq_len; j++) row[j] /= sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + q_head_offset + (int)t_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += row[j] * (v + kv_head_offset + j * head_dim)[d];
        }
        out_row[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Grouped Query Attention (GQA)
// head_idx is the Q head index; KV head = head_idx / (num_heads/num_kv_heads).
// We pass num_kv_heads as an extra constant; group_size = num_heads / num_kv_heads.
// ---------------------------------------------------------------------------

kernel void gqa_forward(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device float*       out        [[buffer(3)]],
    constant int&       seq_len    [[buffer(4)]],
    constant int&       head_dim   [[buffer(5)]],
    constant int&       num_heads  [[buffer(6)]],
    constant int&       num_kv_heads [[buffer(7)]],
    threadgroup float*  smem       [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= seq_len) return;

    int group_size    = num_heads / num_kv_heads;
    int kv_head_idx   = (int)head_idx / group_size;

    int q_head_offset  = (int)head_idx * seq_len * head_dim;
    int kv_head_offset = kv_head_idx   * seq_len * head_dim;
    float scale = rsqrt((float)head_dim);

    const device float* q_row = q + q_head_offset + (int)t_idx * head_dim;

    for (int j = 0; j < seq_len; j++) {
        const device float* k_row = k + kv_head_offset + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
        smem[(int)t_idx * seq_len + j] = dot * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* row = smem + (int)t_idx * seq_len;
    float max_val = row[0];
    for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) { row[j] = exp(row[j] - max_val); sum += row[j]; }
    for (int j = 0; j < seq_len; j++) row[j] /= sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + q_head_offset + (int)t_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += row[j] * (v + kv_head_offset + j * head_dim)[d];
        }
        out_row[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Sliding Window Attention
// Like SDPA but mask positions outside [i-window, i].
// ---------------------------------------------------------------------------

kernel void sliding_window_forward(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    constant int&       seq_len  [[buffer(4)]],
    constant int&       head_dim [[buffer(5)]],
    constant int&       window   [[buffer(6)]],
    threadgroup float*  smem     [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= seq_len) return;

    int i = (int)t_idx;
    int head_offset = (int)head_idx * seq_len * head_dim;
    float scale = rsqrt((float)head_dim);

    const device float* q_row = q + head_offset + i * head_dim;

    for (int j = 0; j < seq_len; j++) {
        float score;
        if (j < i - window || j > i) {
            score = -FLT_MAX;
        } else {
            const device float* k_row = k + head_offset + j * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
            score = dot * scale;
        }
        smem[i * seq_len + j] = score;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* row = smem + i * seq_len;
    float max_val = row[0];
    for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        row[j] = (row[j] == -FLT_MAX) ? 0.0f : exp(row[j] - max_val);
        sum += row[j];
    }
    if (sum > 0.0f) for (int j = 0; j < seq_len; j++) row[j] /= sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + head_offset + i * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += row[j] * (v + head_offset + j * head_dim)[d];
        }
        out_row[d] = acc;
    }
}
