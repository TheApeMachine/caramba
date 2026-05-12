#ifndef METAL_KERNEL_MATH_H
#define METAL_KERNEL_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device/queue and compile all math pipelines.
// metallib_path: absolute path to math.metallib.
// Returns 0 on success, -1 on failure.
int metal_math_init(const char* metallib_path);

// Tiled matrix multiply: C [M*N] = A [M*K] x B [K*N].
// All arrays are row-major float32.
int metal_matmul(const float* A, const float* B, float* C,
                 int M, int K, int N);

// Elementwise add: out[i] = a[i] + b[i].
int metal_add(const float* a, const float* b, float* out, int n);

// Elementwise multiply: out[i] = a[i] * b[i].
int metal_mul(const float* a, const float* b, float* out, int n);

// Resident matmul: A,B,C are MTLBuffer handles (metal_tensor_*). Row-major float32.
// A is [M×K], B is [K×N], C receives [M×N]. Returns 0 on success, -1 on failure.
int metal_matmul_tensor(const void* A, const void* B, void* C, int M, int K, int N);

// Resident elementwise add/mul on length-n float buffers (MTLBuffer handles).
// out[i] = a[i] + b[i] or a[i] * b[i]. Returns 0 on success, -1 on failure.
int metal_add_tensor(const void* a, const void* b, void* out, int n);
int metal_mul_tensor(const void* a, const void* b, void* out, int n);

/*
Resident fused matmul + bias (+ optional GELU).
  A, B, bias — read-only device buffers (row-major float32).
  C — output [M×N] device buffer (written).
  M, K, N — positive matmul dimensions.
  bias_n — bias length: must be N (broadcast per column) or M*N (full).
  gelu — 0 means write matmul+bias only; 1 means apply GELU to each output element.
Returns 0 on success, -1 on invalid arguments or launch failure.
*/
int metal_matmul_add_tensor(
    const void* A, const void* B, const void* bias, void* C,
    int M, int K, int N, int bias_n, int gelu
);

// Scale: dst[i] = src[i] * (1/sqrt(dim)).
int metal_inv_sqrt_dim_scale(const float* src, float* dst, int n, int dim);

// Elementwise exp.
int metal_exp(const float* src, float* dst, int n);

// Elementwise log.
int metal_log(const float* src, float* dst, int n);

// Softmax over last dim. src/dst have num_rows*dim_size elements.
int metal_softmax(const float* src, float* dst, int num_rows, int dim_size);

// Layer norm. src/dst: num_rows*d_model; weight/bias: d_model.
int metal_layernorm(const float* src, float* dst,
                    const float* weight, const float* bias,
                    int num_rows, int d_model, float eps);

// RMS norm. src/dst: num_rows*d_model; weight: d_model.
int metal_rmsnorm(const float* src, float* dst,
                  const float* weight,
                  int num_rows, int d_model, float eps);

// Elementwise sign: dst[i] = +1 if src[i] > 0, -1 if < 0, 0 if == 0.
int metal_sign(const float* src, float* dst, int n);

// Outer product: dst[i*N+j] = a[i] * b[j]. M=len(a), N=len(b).
int metal_outer(const float* a, const float* b, float* dst, int M, int N);

// Optimizer primitives
int metal_axpy(float* dst, const float* src, float scale, int n);
int metal_scale(float* dst, float s, int n);
int metal_sqrt_vec(const float* src, float* dst, int n);
int metal_add_scalar(float* dst, float scalar, int n);
int metal_div_vec(const float* a, const float* b, float* dst, int n);
int metal_clamp_vec(float* dst, float lo, float hi, int n);

#ifdef __cplusplus
}
#endif

#endif /* METAL_KERNEL_MATH_H */
