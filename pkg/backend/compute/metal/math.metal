// Compile with:
// xcrun -sdk macosx metal -c math.metal -o math.air && xcrun -sdk macosx metallib math.air -o math.metallib

#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

// ---------------------------------------------------------------------------
// matmul_kernel: tiled [M,K] x [K,N] -> [M,N]
// Launched as 2D grid: threads_per_group = (TILE_SIZE, TILE_SIZE),
//   groups = (ceil(N/TILE_SIZE), ceil(M/TILE_SIZE))
// Buffers: 0=A, 1=B, 2=C, 3=dims (uint3: M,K,N)
// ---------------------------------------------------------------------------
kernel void matmul_kernel(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device float* C           [[buffer(2)]],
    constant uint3& dims      [[buffer(3)]],
    uint2 gid                 [[thread_position_in_grid]],
    uint2 lid                 [[thread_position_in_threadgroup]],
    uint2 tgid                [[threadgroup_position_in_grid]])
{
    uint M = dims.x, K = dims.y, N = dims.z;
    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;

    threadgroup float tA[TILE_SIZE][TILE_SIZE];
    threadgroup float tB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + lid.x;
        uint bRow = t * TILE_SIZE + lid.y;

        tA[lid.y][lid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tB[lid.y][lid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += tA[lid.y][i] * tB[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

static inline float gelu_value(float x) {
    float x3 = x * x * x;
    float z = 0.7978845608028654f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanh(z));
}

kernel void matmul_add_kernel(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device float* C           [[buffer(3)]],
    constant uint4& dims      [[buffer(4)]],
    uint2 lid                 [[thread_position_in_threadgroup]],
    uint2 tgid                [[threadgroup_position_in_grid]])
{
    uint M = dims.x, K = dims.y, N = dims.z, biasN = dims.w;
    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;

    threadgroup float tA[TILE_SIZE][TILE_SIZE];
    threadgroup float tB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + lid.x;
        uint bRow = t * TILE_SIZE + lid.y;

        tA[lid.y][lid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tB[lid.y][lid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += tA[lid.y][i] * tB[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        acc += biasN == N ? bias[col] : bias[row * N + col];
        C[row * N + col] = acc;
    }
}

kernel void matmul_add_gelu_kernel(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device float* C           [[buffer(3)]],
    constant uint4& dims      [[buffer(4)]],
    uint2 lid                 [[thread_position_in_threadgroup]],
    uint2 tgid                [[threadgroup_position_in_grid]])
{
    uint M = dims.x, K = dims.y, N = dims.z, biasN = dims.w;
    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;

    threadgroup float tA[TILE_SIZE][TILE_SIZE];
    threadgroup float tB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + lid.x;
        uint bRow = t * TILE_SIZE + lid.y;

        tA[lid.y][lid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tB[lid.y][lid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; i++) {
            acc += tA[lid.y][i] * tB[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        acc += biasN == N ? bias[col] : bias[row * N + col];
        C[row * N + col] = gelu_value(acc);
    }
}

// ---------------------------------------------------------------------------
// add_kernel: elementwise a + b
// ---------------------------------------------------------------------------
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint i                [[thread_position_in_grid]])
{
    out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// mul_kernel: elementwise a * b
// ---------------------------------------------------------------------------
kernel void mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint i                [[thread_position_in_grid]])
{
    out[i] = a[i] * b[i];
}

// ---------------------------------------------------------------------------
// inv_sqrt_dim_scale_kernel: x * (1/sqrt(dim))
// Buffers: 0=src, 1=dst, 2=scale (float scalar)
// ---------------------------------------------------------------------------
kernel void inv_sqrt_dim_scale_kernel(
    device const float* src   [[buffer(0)]],
    device float* dst         [[buffer(1)]],
    constant float& scale     [[buffer(2)]],
    uint i                    [[thread_position_in_grid]])
{
    dst[i] = src[i] * scale;
}

// ---------------------------------------------------------------------------
// exp_kernel: elementwise exp(x)
// ---------------------------------------------------------------------------
kernel void exp_kernel(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = exp(src[i]);
}

// ---------------------------------------------------------------------------
// log_kernel: elementwise log(x)
// ---------------------------------------------------------------------------
kernel void log_kernel(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = log(src[i]);
}

// ---------------------------------------------------------------------------
// softmax_kernel: one threadgroup per row; reduction over dim_size elements
// Buffers: 0=src, 1=dst, 2=dim_size (uint)
// Grid: (num_rows, 1), threadgroup: (min(dim_size, 256), 1)
// ---------------------------------------------------------------------------
kernel void softmax_kernel(
    device const float* src   [[buffer(0)]],
    device float* dst         [[buffer(1)]],
    constant uint& dim_size   [[buffer(2)]],
    uint row                  [[threadgroup_position_in_grid]],
    uint lid                  [[thread_position_in_threadgroup]],
    uint tg_size              [[threads_per_threadgroup]])
{
    threadgroup float smem[256];
    uint offset = row * dim_size;

    // Phase 1: reduce max
    float lmax = -INFINITY;
    for (uint i = lid; i < dim_size; i += tg_size) {
        lmax = max(lmax, src[offset + i]);
    }
    smem[lid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] = max(smem[lid], smem[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gmax = smem[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: compute exp and partial sum
    float lsum = 0.0f;
    for (uint i = lid; i < dim_size; i += tg_size) {
        float e = exp(src[offset + i] - gmax);
        dst[offset + i] = e;
        lsum += e;
    }
    smem[lid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gsum = smem[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: normalize
    for (uint i = lid; i < dim_size; i += tg_size) {
        dst[offset + i] /= gsum;
    }
}

// ---------------------------------------------------------------------------
// layernorm_kernel: one threadgroup per row
// Buffers: 0=src, 1=dst, 2=weight(gamma), 3=bias(beta),
//          4=d_model (uint), 5=eps (float)
// ---------------------------------------------------------------------------
kernel void layernorm_kernel(
    device const float* src     [[buffer(0)]],
    device float* dst           [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    device const float* bias    [[buffer(3)]],
    constant uint& d_model      [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint lid                    [[thread_position_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]])
{
    threadgroup float smem[256];
    uint offset = row * d_model;

    // mean
    float lsum = 0.0f;
    for (uint i = lid; i < d_model; i += tg_size) lsum += src[offset + i];
    smem[lid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = smem[0] / (float)d_model;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // variance
    float lvar = 0.0f;
    for (uint i = lid; i < d_model; i += tg_size) {
        float diff = src[offset + i] - mean;
        lvar += diff * diff;
    }
    smem[lid] = lvar;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(smem[0] / (float)d_model + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // normalize + affine
    for (uint i = lid; i < d_model; i += tg_size) {
        dst[offset + i] = (src[offset + i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// ---------------------------------------------------------------------------
// rmsnorm_kernel: one threadgroup per row
// Buffers: 0=src, 1=dst, 2=weight, 3=d_model (uint), 4=eps (float)
// ---------------------------------------------------------------------------
kernel void rmsnorm_kernel(
    device const float* src     [[buffer(0)]],
    device float* dst           [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    constant uint& d_model      [[buffer(3)]],
    constant float& eps         [[buffer(4)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint lid                    [[thread_position_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]])
{
    threadgroup float smem[256];
    uint offset = row * d_model;

    float lss = 0.0f;
    for (uint i = lid; i < d_model; i += tg_size) {
        float v = src[offset + i];
        lss += v * v;
    }
    smem[lid] = lss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(smem[0] / (float)d_model + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < d_model; i += tg_size) {
        dst[offset + i] = src[offset + i] * inv_rms * weight[i];
    }
}

// ---------------------------------------------------------------------------
// sign_kernel: out[i] = sign(src[i])
// Buffers: 0=src, 1=dst; thread_position_in_grid = i
// ---------------------------------------------------------------------------
kernel void sign_kernel(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    float v = src[i];
    dst[i] = (v > 0.0f) ? 1.0f : (v < 0.0f) ? -1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// outer_kernel: dst[row*N+col] = a[row] * b[col]
// Buffers: 0=a, 1=b, 2=dst, 3=dims (uint2: M, N)
// Launch as 2D: grid=(N,M), threadgroup=(1,1) or tuned
// ---------------------------------------------------------------------------
kernel void outer_kernel(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device float* dst      [[buffer(2)]],
    constant uint2& dims   [[buffer(3)]],
    uint2 gid              [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    if (row < dims.x && col < dims.y)
        dst[row * dims.y + col] = a[row] * b[col];
}

// ---------------------------------------------------------------------------
// Optimizer kernels
// ---------------------------------------------------------------------------

kernel void axpy_kernel(
    device float* dst       [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant float& scale   [[buffer(2)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] += scale * src[i];
}

kernel void scale_kernel2(
    device float* dst     [[buffer(0)]],
    constant float& s     [[buffer(1)]],
    uint i                [[thread_position_in_grid]])
{
    dst[i] *= s;
}

kernel void sqrt_vec_kernel(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] = sqrt(src[i]);
}

kernel void add_scalar_kernel(
    device float* dst       [[buffer(0)]],
    constant float& scalar  [[buffer(1)]],
    uint i                  [[thread_position_in_grid]])
{
    dst[i] += scalar;
}

kernel void div_vec_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* dst     [[buffer(2)]],
    uint i                [[thread_position_in_grid]])
{
    dst[i] = a[i] / b[i];
}

kernel void clamp_vec_kernel(
    device float* dst   [[buffer(0)]],
    constant float& lo  [[buffer(1)]],
    constant float& hi  [[buffer(2)]],
    uint i              [[thread_position_in_grid]])
{
    dst[i] = clamp(dst[i], lo, hi);
}
