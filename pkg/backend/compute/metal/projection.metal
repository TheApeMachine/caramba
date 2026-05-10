// Compile with:
// xcrun -sdk macosx metal -c projection.metal -o projection.air && xcrun -sdk macosx metallib projection.air -o projection.metallib

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Tiled matmul: C[M,N] = A[M,K] @ B[K,N]  (all row-major, float32)
// threadgroup TILE=16: each thread computes one C[i,j].
// ---------------------------------------------------------------------------

#define TILE 16

kernel void matmul_kernel(
    device const float* A     [[buffer(0)]],  // [M*K]
    device const float* B     [[buffer(1)]],  // [K*N]
    device       float* C     [[buffer(2)]],  // [M*N]
    constant     uint&  M_dim [[buffer(3)]],
    constant     uint&  K_dim [[buffer(4)]],
    constant     uint&  N_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M_dim || col >= N_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K_dim; k++) {
        acc += A[row * K_dim + k] * B[k * N_dim + col];
    }
    C[row * N_dim + col] = acc;
}

// ---------------------------------------------------------------------------
// linear_kernel: C = A @ W^T + bias
//   A[M,K], W[N,K] (each row of W is one output neuron's weight vector)
//   bias[N] (optional; pass NULL / skip via sentinel)
// ---------------------------------------------------------------------------
kernel void linear_kernel(
    device const float* A     [[buffer(0)]],  // [M*K]
    device const float* W     [[buffer(1)]],  // [N*K]  (W^T implied)
    device const float* bias  [[buffer(2)]],  // [N] or zero-length
    device       float* C     [[buffer(3)]],  // [M*N]
    constant     uint&  M_dim [[buffer(4)]],
    constant     uint&  K_dim [[buffer(5)]],
    constant     uint&  N_dim [[buffer(6)]],
    constant     uint&  has_bias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M_dim || col >= N_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K_dim; k++) {
        acc += A[row * K_dim + k] * W[col * K_dim + k];  // W^T
    }
    if (has_bias) {
        acc += bias[col];
    }
    C[row * N_dim + col] = acc;
}

// ---------------------------------------------------------------------------
// fused_qkv_kernel: same as linear_kernel — output split is done in Go.
// ---------------------------------------------------------------------------
kernel void fused_qkv_kernel(
    device const float* A     [[buffer(0)]],
    device const float* W     [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device       float* C     [[buffer(3)]],
    constant     uint&  M_dim [[buffer(4)]],
    constant     uint&  K_dim [[buffer(5)]],
    constant     uint&  N_dim [[buffer(6)]],
    constant     uint&  has_bias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M_dim || col >= N_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K_dim; k++) {
        acc += A[row * K_dim + k] * W[col * K_dim + k];
    }
    if (has_bias) {
        acc += bias[col];
    }
    C[row * N_dim + col] = acc;
}

// ---------------------------------------------------------------------------
// tied_embedding_kernel: logits = hidden @ embed_weight^T
//   hidden[M,D], embed_weight[V,D] → logits[M,V]
// ---------------------------------------------------------------------------
kernel void tied_embedding_kernel(
    device const float* hidden  [[buffer(0)]],  // [M*D]
    device const float* weight  [[buffer(1)]],  // [V*D]
    device       float* logits  [[buffer(2)]],  // [M*V]
    constant     uint&  M_dim   [[buffer(3)]],
    constant     uint&  D_dim   [[buffer(4)]],
    constant     uint&  V_dim   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // token index (0..M-1)
    uint col = gid.x;  // vocab index (0..V-1)
    if (row >= M_dim || col >= V_dim) return;

    float acc = 0.0f;
    for (uint d = 0; d < D_dim; d++) {
        acc += hidden[row * D_dim + d] * weight[col * D_dim + d];
    }
    logits[row * V_dim + col] = acc;
}
