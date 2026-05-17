// Compile with:
// xcrun -sdk macosx metal -c physics.metal -o physics.air && \
//   xcrun -sdk macosx metallib physics.air -o physics.metallib

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 1D periodic Laplacian
//   dst[i] = (src[(i-1+n)%n] + src[(i+1)%n] - 2*src[i]) * inv_h2
// Each thread handles one output element.
// ---------------------------------------------------------------------------
kernel void laplacian_1d_kernel(
    device const float* src    [[buffer(0)]],
    device float*       dst    [[buffer(1)]],
    constant int&       n      [[buffer(2)]],
    constant float&     inv_h2 [[buffer(3)]],
    uint                idx    [[thread_position_in_grid]])
{
    int i = (int)idx;
    if (i >= n) return;

    int left  = (i == 0)     ? n - 1 : i - 1;
    int right = (i == n - 1) ? 0     : i + 1;

    dst[i] = (src[left] + src[right] - 2.0f * src[i]) * inv_h2;
}

// ---------------------------------------------------------------------------
// 2D periodic 5-point Laplacian on row-major [H, W]
// ---------------------------------------------------------------------------
kernel void laplacian_2d_kernel(
    device const float* src    [[buffer(0)]],
    device float*       dst    [[buffer(1)]],
    constant int&       H      [[buffer(2)]],
    constant int&       W      [[buffer(3)]],
    constant float&     inv_h2 [[buffer(4)]],
    uint                idx    [[thread_position_in_grid]])
{
    int total = H * W;
    int linear = (int)idx;
    if (linear >= total) return;

    int i = linear / W;
    int j = linear - i * W;

    int up    = (i == 0)     ? H - 1 : i - 1;
    int down  = (i == H - 1) ? 0     : i + 1;
    int left  = (j == 0)     ? W - 1 : j - 1;
    int right = (j == W - 1) ? 0     : j + 1;

    float center = src[linear];
    float horizontal = src[i * W + left] + src[i * W + right];
    float vertical   = src[up * W + j]   + src[down * W + j];

    dst[linear] = (horizontal + vertical - 4.0f * center) * inv_h2;
}

// ---------------------------------------------------------------------------
// 3D periodic 7-point Laplacian on row-major [D, H, W]
// ---------------------------------------------------------------------------
kernel void laplacian_3d_kernel(
    device const float* src    [[buffer(0)]],
    device float*       dst    [[buffer(1)]],
    constant int&       D      [[buffer(2)]],
    constant int&       H      [[buffer(3)]],
    constant int&       W      [[buffer(4)]],
    constant float&     inv_h2 [[buffer(5)]],
    uint                idx    [[thread_position_in_grid]])
{
    int hw = H * W;
    int total = D * hw;
    int linear = (int)idx;
    if (linear >= total) return;

    int k = linear / hw;
    int rem = linear - k * hw;
    int i = rem / W;
    int j = rem - i * W;

    int front = (k == 0)     ? D - 1 : k - 1;
    int back  = (k == D - 1) ? 0     : k + 1;
    int up    = (i == 0)     ? H - 1 : i - 1;
    int down  = (i == H - 1) ? 0     : i + 1;
    int left  = (j == 0)     ? W - 1 : j - 1;
    int right = (j == W - 1) ? 0     : j + 1;

    float center = src[linear];
    float horizontal = src[k * hw + i * W + left] + src[k * hw + i * W + right];
    float vertical   = src[k * hw + up * W + j]   + src[k * hw + down * W + j];
    float transverse = src[front * hw + i * W + j] + src[back * hw + i * W + j];

    dst[linear] = (horizontal + vertical + transverse - 6.0f * center) * inv_h2;
}
