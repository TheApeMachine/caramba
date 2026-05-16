// Compile with:
// xcrun -sdk macosx metal -c masking.metal -o masking.air && xcrun -sdk macosx metallib masking.air -o masking.metallib

#include <metal_stdlib>
using namespace metal;

static inline float negative_infinity() {
    return as_type<float>(0xff800000u);
}

// causal_mask_kernel: generates a causal (lower-triangular) attention mask.
// idx.x = column, idx.y = row
// out[row * seq_len + col] = 0.0 if col <= row, else -Inf
kernel void causal_mask_kernel(
    device float*       out      [[buffer(0)]],
    constant int&       seq_len  [[buffer(1)]],
    uint2               idx      [[thread_position_in_grid]])
{
    uint row = idx.y;
    uint col = idx.x;
    if (row >= (uint)seq_len || col >= (uint)seq_len) return;
    out[row * seq_len + col] = (col <= row) ? 0.0f : negative_infinity();
}

// apply_mask_kernel: elementwise add scores + mask -> out
kernel void apply_mask_kernel(
    device const float* scores [[buffer(0)]],
    device const float* mask   [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    uint                i      [[thread_position_in_grid]])
{
    out[i] = scores[i] + mask[i];
}
