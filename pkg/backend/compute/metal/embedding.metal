// Compile with:
// xcrun -sdk macosx metal -c embedding.metal -o embedding.air && xcrun -sdk macosx metallib embedding.air -o embedding.metallib

#include <metal_stdlib>
using namespace metal;

// token_embedding_kernel
// Dispatched with total_threads = batch*seq_len*d_model.
// Each thread copies one float from the weight table into the output buffer.
kernel void token_embedding_kernel(
    device const float* weight  [[buffer(0)]],
    device const float* tokens  [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant int&       d_model [[buffer(3)]],
    uint idx                    [[thread_position_in_grid]])
{
    int token_idx = (int)idx / d_model;
    int dim_idx   = (int)idx % d_model;
    int token_id  = (int)tokens[token_idx];
    out[idx] = weight[token_id * d_model + dim_idx];
}
