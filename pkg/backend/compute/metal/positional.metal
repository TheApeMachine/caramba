// Compile with:
// xcrun -sdk macosx metal -c positional.metal -o positional.air && xcrun -sdk macosx metallib positional.air -o positional.metallib

#include <metal_stdlib>
using namespace metal;

// RoPE kernel.
// Each thread handles one (batch*head, position) slot, rotating one pair
// of dimensions (2i, 2i+1) using precomputed cos/sin tables.
//
// Layout:
//   x, out: [total_heads * seq_len * head_dim]
//   cos_table, sin_table: [seq_len * num_pairs]
//   idx encodes: slot = b_h * seq_len * num_pairs + t * num_pairs + i
//                but we dispatch total = total_heads * seq_len * num_pairs threads
//
// thread idx = (b_h * seq_len + t) * num_pairs + i
kernel void rope_kernel(
    device const float* x         [[buffer(0)]],
    device float*       out       [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant int&       seq_len   [[buffer(4)]],
    constant int&       head_dim  [[buffer(5)]],
    constant int&       rope_mode [[buffer(6)]],
    uint                idx       [[thread_position_in_grid]])
{
    int num_pairs   = head_dim / 2;
    int pair_idx    = (int)(idx % (uint)num_pairs);          // i
    int slot        = (int)(idx / (uint)num_pairs);          // b_h * seq_len + t
    int t           = slot % seq_len;
    // int b_h      = slot / seq_len;  (not needed separately)

    int slot_base = slot * head_dim;
    int first_idx = slot_base + pair_idx * 2;
    int second_idx = first_idx + 1;

    if (rope_mode == 1) {
        first_idx = slot_base + pair_idx;
        second_idx = first_idx + num_pairs;
    }

    float x0 = x[first_idx];
    float x1 = x[second_idx];

    int tbl_idx = t * num_pairs + pair_idx;
    float c = cos_table[tbl_idx];
    float s = sin_table[tbl_idx];

    out[first_idx]  = x0 * c - x1 * s;
    out[second_idx] = x0 * s + x1 * c;
}

// ALiBi kernel.
// Each thread computes one output element out[h * seq_q * seq_k + q * seq_k + k]
// = slopes[h] * (k - q)
//
// idx = h * seq_len_q * seq_len_k + q * seq_len_k + k
kernel void alibi_kernel(
    device float*       out       [[buffer(0)]],
    device const float* slopes    [[buffer(1)]],
    constant int&       seq_len_q [[buffer(2)]],
    constant int&       seq_len_k [[buffer(3)]],
    constant int&       causal    [[buffer(4)]],
    uint                idx       [[thread_position_in_grid]])
{
    int seqK   = seq_len_k;
    int seqQ   = seq_len_q;
    int total  = seqQ * seqK;
    int h      = (int)(idx / (uint)total);
    int rem    = (int)(idx % (uint)total);
    int q      = rem / seqK;
    int k      = rem % seqK;
    float distance = (float)(k - q);

    if (causal == 0 && distance < 0.0f) {
        distance = -distance;
    }

    out[idx] = slopes[h] * distance;
}
