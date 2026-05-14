// Compile with:
// xcrun -sdk macosx metal -c math.metal -o math.air && xcrun -sdk macosx metallib math.air -o math.metallib

#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

// Shared tiled matmul accumulation (K dimension tiled).
static inline float compute_matmul_tile_accum(
    device const float* A,
    device const float* B,
    uint M, uint K, uint N,
    uint row, uint col,
    uint2 lid,
    threadgroup float tA[TILE_SIZE][TILE_SIZE],
    threadgroup float tB[TILE_SIZE][TILE_SIZE])
{
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

    return acc;
}

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

    float acc = compute_matmul_tile_accum(A, B, M, K, N, row, col, lid, tA, tB);

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

    float acc = compute_matmul_tile_accum(A, B, M, K, N, row, col, lid, tA, tB);

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

    float acc = compute_matmul_tile_accum(A, B, M, K, N, row, col, lid, tA, tB);

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

kernel void logsumexp_kernel(
    device const float* src   [[buffer(0)]],
    device float* dst         [[buffer(1)]],
    constant uint& dim_size   [[buffer(2)]],
    uint row                  [[threadgroup_position_in_grid]],
    uint lid                  [[thread_position_in_threadgroup]],
    uint tg_size              [[threads_per_threadgroup]])
{
    threadgroup float smem[256];
    uint offset = row * dim_size;

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

    float lsum = 0.0f;
    for (uint i = lid; i < dim_size; i += tg_size) {
        lsum += exp(src[offset + i] - gmax);
    }
    smem[lid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        dst[row] = gmax + log(smem[0]);
    }
}

static inline ulong dropout_mix64(ulong value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9UL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebUL;
    value ^= value >> 31;
    return value;
}

kernel void dropout_kernel(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant float& p       [[buffer(2)]],
    constant uint& training [[buffer(3)]],
    constant uint& seed     [[buffer(4)]],
    uint i                  [[thread_position_in_grid]])
{
    if (training == 0 || p == 0.0f) {
        dst[i] = src[i];
        return;
    }

    ulong mixed = dropout_mix64((ulong(seed) << 32) ^ ulong(i));
    float unit = float(mixed >> 11) * (1.0f / 9007199254740992.0f);

    dst[i] = unit >= p ? src[i] / (1.0f - p) : 0.0f;
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

kernel void train_mse_loss_kernel(
    device const float* predictions [[buffer(0)]],
    device const float* targets     [[buffer(1)]],
    device atomic_float* out        [[buffer(2)]],
    constant uint& n                [[buffer(3)]],
    uint i                          [[thread_position_in_grid]])
{
    if (i >= n) return;
    float diff = predictions[i] - targets[i];
    atomic_fetch_add_explicit(out, diff * diff / float(n), memory_order_relaxed);
}

kernel void train_mse_grad_kernel(
    device const float* predictions [[buffer(0)]],
    device const float* targets     [[buffer(1)]],
    device float* out               [[buffer(2)]],
    constant uint& n                [[buffer(3)]],
    uint i                          [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = 2.0f * (predictions[i] - targets[i]) / float(n);
}

kernel void train_ce_stats_kernel(
    device const float* logits [[buffer(0)]],
    device float* stats        [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]])
{
    if (gid != 0) return;

    float max_value = -INFINITY;
    for (uint i = 0; i < n; i++) {
        max_value = max(max_value, logits[i]);
    }

    float sum_value = 0.0f;
    for (uint i = 0; i < n; i++) {
        sum_value += exp(logits[i] - max_value);
    }

    stats[0] = max_value;
    stats[1] = sum_value;
}

kernel void train_cross_entropy_loss_kernel(
    device const float* logits  [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device atomic_float* out    [[buffer(2)]],
    device const float* stats   [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint i                      [[thread_position_in_grid]])
{
    if (i >= n) return;
    float probability = exp(logits[i] - stats[0]) / stats[1];
    atomic_fetch_add_explicit(
        out, -log(probability + 1.0e-9f) * targets[i], memory_order_relaxed);
}

kernel void train_cross_entropy_grad_kernel(
    device const float* logits  [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* out           [[buffer(2)]],
    device const float* stats   [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint i                      [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = exp(logits[i] - stats[0]) / stats[1] - targets[i];
}

kernel void bench_accuracy_kernel(
    device const float* predictions [[buffer(0)]],
    device const float* targets     [[buffer(1)]],
    device float* out               [[buffer(2)]],
    constant uint& n                [[buffer(3)]],
    uint lid                        [[thread_position_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]])
{
    threadgroup float pred_values[256];
    threadgroup float target_values[256];
    threadgroup int pred_indices[256];
    threadgroup int target_indices[256];

    float pred_best = -INFINITY;
    float target_best = -INFINITY;
    int pred_index = 0;
    int target_index = 0;

    for (uint i = lid; i < n; i += tg_size) {
        if (predictions[i] > pred_best) {
            pred_best = predictions[i];
            pred_index = int(i);
        }
        if (targets[i] > target_best) {
            target_best = targets[i];
            target_index = int(i);
        }
    }

    pred_values[lid] = pred_best;
    target_values[lid] = target_best;
    pred_indices[lid] = pred_index;
    target_indices[lid] = target_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (pred_values[lid + stride] > pred_values[lid]) {
                pred_values[lid] = pred_values[lid + stride];
                pred_indices[lid] = pred_indices[lid + stride];
            }
            if (target_values[lid + stride] > target_values[lid]) {
                target_values[lid] = target_values[lid + stride];
                target_indices[lid] = target_indices[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        out[0] = pred_indices[0] == target_indices[0] ? 1.0f : 0.0f;
    }
}

kernel void bench_f1_counts_kernel(
    device const float* predictions [[buffer(0)]],
    device const float* targets     [[buffer(1)]],
    device atomic_float* out        [[buffer(2)]],
    constant uint& n                [[buffer(3)]],
    uint i                          [[thread_position_in_grid]])
{
    if (i >= n) return;
    bool predicted = predictions[i] >= 0.5f;
    bool actual = targets[i] >= 0.5f;

    if (predicted && actual) {
        atomic_fetch_add_explicit(out, 1.0f, memory_order_relaxed);
    }
    if (predicted && !actual) {
        atomic_fetch_add_explicit(out + 1, 1.0f, memory_order_relaxed);
    }
    if (!predicted && actual) {
        atomic_fetch_add_explicit(out + 2, 1.0f, memory_order_relaxed);
    }
}
