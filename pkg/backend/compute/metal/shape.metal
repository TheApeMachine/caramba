// Compile with:
// xcrun -sdk macosx metal -c shape.metal -o shape.air && xcrun -sdk macosx metallib shape.air -o shape.metallib

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// transpose_kernel
// Swaps two dimensions (dim0, dim1) of an N-D tensor stored row-major.
// shape[0..rank-1] holds the input tensor dimensions.
// Each thread handles one output element.
// ---------------------------------------------------------------------------
kernel void transpose_kernel(
    device const float*  src    [[buffer(0)]],
    device float*        dst    [[buffer(1)]],
    constant int*        shape  [[buffer(2)]],
    constant int&        rank   [[buffer(3)]],
    constant int&        dim0   [[buffer(4)]],
    constant int&        dim1   [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    // Decode flat index idx into coords using input strides (row-major).
    // Max supported rank = 8.
    int coords[8];
    int strides[8];

    // Compute input strides.
    strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    int rem = (int)idx;
    for (int d = 0; d < rank; d++) {
        coords[d] = rem / strides[d];
        rem       = rem % strides[d];
    }

    // Swap the two dimensions.
    int tmp         = coords[dim0];
    coords[dim0]    = coords[dim1];
    coords[dim1]    = tmp;

    // Compute output shape and strides after swap.
    int outShape[8];
    for (int d = 0; d < rank; d++) outShape[d] = shape[d];
    outShape[dim0] = shape[dim1];
    outShape[dim1] = shape[dim0];

    int outStrides[8];
    outStrides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--) {
        outStrides[d] = outStrides[d + 1] * outShape[d + 1];
    }

    int outIdx = 0;
    for (int d = 0; d < rank; d++) {
        outIdx += coords[d] * outStrides[d];
    }

    dst[outIdx] = src[idx];
}

// ---------------------------------------------------------------------------
// copy_kernel — elementwise copy (reshape: data is identical, shape differs).
// ---------------------------------------------------------------------------
kernel void copy_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// concat_kernel — concatenates two tensors along a single split point.
// Each thread writes one output element.
// splitPoint is the number of elements in the first tensor.
// ---------------------------------------------------------------------------
kernel void concat_kernel(
    device const float* srcA       [[buffer(0)]],
    device const float* srcB       [[buffer(1)]],
    device float*       dst        [[buffer(2)]],
    constant int&       splitPoint [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i < splitPoint) {
        dst[i] = srcA[i];
    } else {
        dst[i] = srcB[i - splitPoint];
    }
}

kernel void split_kernel(
    device const float* src       [[buffer(0)]],
    device float*       dst       [[buffer(1)]],
    constant int&       outer     [[buffer(2)]],
    constant int&       dimSize   [[buffer(3)]],
    constant int&       splitSize [[buffer(4)]],
    constant int&       inner     [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    int elementInChunk = splitSize * inner;
    int chunkElements = outer * elementInChunk;
    int chunk = int(i) / chunkElements;
    int withinChunk = int(i) - chunk * chunkElements;
    int outerIndex = withinChunk / elementInChunk;
    int within = withinChunk - outerIndex * elementInChunk;
    int srcIndex = (outerIndex * dimSize + chunk * splitSize) * inner + within;

    dst[i] = src[srcIndex];
}

// ---------------------------------------------------------------------------
// view_as_heads_kernel
// Transposes dims 1 and 2 of a [B, T, H, head_dim] tensor ->
// [B, H, T, head_dim].
// shape = {B, T, H, head_dim}
// ---------------------------------------------------------------------------
kernel void view_as_heads_kernel(
    device const float* src      [[buffer(0)]],
    device float*       dst      [[buffer(1)]],
    constant int&       B        [[buffer(2)]],
    constant int&       T        [[buffer(3)]],
    constant int&       H        [[buffer(4)]],
    constant int&       head_dim [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    // Decode [B, T, H, head_dim] coords.
    int hd   = (int)idx % head_dim;
    int rem  = (int)idx / head_dim;
    int h    = rem % H;
    rem      = rem / H;
    int t    = rem % T;
    int b    = rem / T;

    // Output layout [B, H, T, head_dim] -> coord (b, h, t, hd).
    int outIdx = b * (H * T * head_dim)
               + h * (T * head_dim)
               + t * head_dim
               + hd;

    dst[outIdx] = src[idx];
}

// ---------------------------------------------------------------------------
// merge_heads_kernel
// Transposes dims 1 and 2 of a [B, H, T, head_dim] tensor ->
// [B, T, H, head_dim].
// ---------------------------------------------------------------------------
kernel void merge_heads_kernel(
    device const float* src      [[buffer(0)]],
    device float*       dst      [[buffer(1)]],
    constant int&       B        [[buffer(2)]],
    constant int&       H        [[buffer(3)]],
    constant int&       T        [[buffer(4)]],
    constant int&       head_dim [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    // Decode [B, H, T, head_dim] coords.
    int hd   = (int)idx % head_dim;
    int rem  = (int)idx / head_dim;
    int t    = rem % T;
    rem      = rem / T;
    int h    = rem % H;
    int b    = rem / H;

    // Output layout [B, T, H, head_dim] -> coord (b, t, h, hd).
    int outIdx = b * (T * H * head_dim)
               + t * (H * head_dim)
               + h * head_dim
               + hd;

    dst[outIdx] = src[idx];
}

kernel void last_token_kernel(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant int&       seq_len [[buffer(2)]],
    constant int&       feature [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    int outer = (int)idx / feature;
    int feature_idx = (int)idx % feature;
    int src_idx = (outer * seq_len + (seq_len - 1)) * feature + feature_idx;

    dst[idx] = src[src_idx];
}
