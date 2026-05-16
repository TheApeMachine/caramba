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
    constant int&       query_len [[buffer(4)]],
    constant int&       key_value_len [[buffer(5)]],
    constant int&       key_value_stride [[buffer(6)]],
    constant int&       head_dim [[buffer(7)]],
    constant int&       causal [[buffer(8)]],
    threadgroup float*  smem     [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= query_len) return;

    int q_head_offset = (int)head_idx * query_len * head_dim;
    int kv_head_offset = (int)head_idx * key_value_stride * head_dim;
    float scale = rsqrt((float)head_dim);
    int visible = key_value_len;
    if (causal != 0) {
        int offset = key_value_len - query_len;
        visible = offset + (int)t_idx + 1;
    }

    // Compute scores[t_idx, j] = dot(Q[t_idx], K[j]) * scale
    const device float* q_row = q + q_head_offset + (int)t_idx * head_dim;

    for (int j = 0; j < visible; j++) {
        const device float* k_row = k + kv_head_offset + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        smem[(int)t_idx * key_value_len + j] = dot * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax over row t_idx
    threadgroup float* row = smem + (int)t_idx * key_value_len;
    float max_val = row[0];
    for (int j = 1; j < visible; j++) max_val = max(max_val, row[j]);

    float sum = 0.0f;
    for (int j = 0; j < visible; j++) {
        row[j] = exp(row[j] - max_val);
        sum += row[j];
    }
    for (int j = 0; j < visible; j++) row[j] /= sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output[t_idx] = sum_j weights[j] * V[j]
    device float* out_row = out + q_head_offset + (int)t_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < visible; j++) {
            acc += row[j] * (v + kv_head_offset + j * head_dim)[d];
        }
        out_row[d] = acc;
    }
}

kernel void kv_append_forward(
    device const float* previous_key   [[buffer(0)]],
    device const float* previous_value [[buffer(1)]],
    device const float* chunk_key      [[buffer(2)]],
    device const float* chunk_value    [[buffer(3)]],
    device float*       output_key     [[buffer(4)]],
    device float*       output_value   [[buffer(5)]],
    constant int&       previous_len   [[buffer(6)]],
    constant int&       chunk_len      [[buffer(7)]],
    constant int&       head_dim       [[buffer(8)]],
    constant int&       total_len      [[buffer(9)]],
    uint index [[thread_position_in_grid]])
{
    int per_previous = previous_len * head_dim;
    int per_chunk = chunk_len * head_dim;
    int per_total = total_len * head_dim;
    int head_index = (int)index / per_total;
    int offset = (int)index - head_index * per_total;

    if (offset < per_previous) {
        int previous_index = head_index * per_previous + offset;
        output_key[index] = previous_key[previous_index];
        output_value[index] = previous_value[previous_index];
        return;
    }

    int chunk_offset = offset - per_previous;

    if (chunk_offset < per_chunk) {
        int chunk_index = head_index * per_chunk + chunk_offset;
        output_key[index] = chunk_key[chunk_index];
        output_value[index] = chunk_value[chunk_index];
    }
}

kernel void kv_repack_forward(
    device const float* previous_key   [[buffer(0)]],
    device const float* previous_value [[buffer(1)]],
    device float*       output_key     [[buffer(2)]],
    device float*       output_value   [[buffer(3)]],
    constant int&       current_len    [[buffer(4)]],
    constant int&       head_dim       [[buffer(5)]],
    constant int&       previous_capacity [[buffer(6)]],
    constant int&       output_capacity [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    int per_current = current_len * head_dim;
    int head_index = (int)index / per_current;
    int offset = (int)index - head_index * per_current;
    int token = offset / head_dim;
    int dim = offset - token * head_dim;
    int previous_index = (head_index * previous_capacity + token) * head_dim + dim;
    int output_index = (head_index * output_capacity + token) * head_dim + dim;

    output_key[output_index] = previous_key[previous_index];
    output_value[output_index] = previous_value[previous_index];
}

kernel void kv_write_forward(
    device float*       cache_key   [[buffer(0)]],
    device float*       cache_value [[buffer(1)]],
    device const float* chunk_key   [[buffer(2)]],
    device const float* chunk_value [[buffer(3)]],
    constant int&       start_len   [[buffer(4)]],
    constant int&       chunk_len   [[buffer(5)]],
    constant int&       head_dim    [[buffer(6)]],
    constant int&       capacity    [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    int per_chunk = chunk_len * head_dim;
    int head_index = (int)index / per_chunk;
    int offset = (int)index - head_index * per_chunk;
    int token = offset / head_dim;
    int dim = offset - token * head_dim;
    int cache_index = (head_index * capacity + start_len + token) * head_dim + dim;

    cache_key[cache_index] = chunk_key[index];
    cache_value[cache_index] = chunk_value[index];
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
    constant int&       num_heads [[buffer(6)]],
    threadgroup float*  smem     [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= seq_len) return;

    int batch_idx      = (int)head_idx / num_heads;
    int q_head_offset  = (int)head_idx * seq_len * head_dim;
    int kv_head_offset = batch_idx * seq_len * head_dim; // one KV head per batch item
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
    constant int&       query_len  [[buffer(4)]],
    constant int&       key_value_len [[buffer(5)]],
    constant int&       key_value_stride [[buffer(6)]],
    constant int&       head_dim   [[buffer(7)]],
    constant int&       num_heads  [[buffer(8)]],
    constant int&       num_kv_heads [[buffer(9)]],
    constant int&       causal     [[buffer(10)]],
    threadgroup float*  smem       [[threadgroup(0)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint t_idx    [[thread_position_in_threadgroup]])
{
    if ((int)t_idx >= query_len) return;

    int group_size         = num_heads / num_kv_heads;
    int batch_idx          = (int)head_idx / num_heads;
    int head_within_batch  = (int)head_idx % num_heads;
    int kv_head_idx        = head_within_batch / group_size;

    int q_head_offset  = (int)head_idx * query_len * head_dim;
    int kv_head_offset = (batch_idx * num_kv_heads + kv_head_idx) *
                         key_value_stride * head_dim;
    float scale = rsqrt((float)head_dim);
    int visible = key_value_len;

    if (causal != 0) {
        int offset = key_value_len - query_len;
        visible = offset + (int)t_idx + 1;
    }

    const device float* q_row = q + q_head_offset + (int)t_idx * head_dim;

    for (int j = 0; j < visible; j++) {
        const device float* k_row = k + kv_head_offset + j * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
        smem[(int)t_idx * key_value_len + j] = dot * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* row = smem + (int)t_idx * key_value_len;
    float max_val = row[0];
    for (int j = 1; j < visible; j++) max_val = max(max_val, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < visible; j++) { row[j] = exp(row[j] - max_val); sum += row[j]; }
    for (int j = 0; j < visible; j++) row[j] /= sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + q_head_offset + (int)t_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < visible; j++) {
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
