#ifndef METAL_KERNEL_MATH_H
#define METAL_KERNEL_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device/queue and compile all math pipelines.
// metallib_path: absolute path to math.metallib.
// Returns 0 on success, -1 on failure.
int metal_math_init(const char* metallib_path);
int metal_math_shutdown(void);

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

int metal_inv_sqrt_dim_scale_tensor(const void* src, void* dst, int n, int dim);
int metal_exp_tensor(const void* src, void* dst, int n);
int metal_log_tensor(const void* src, void* dst, int n);
int metal_sign_tensor(const void* src, void* dst, int n);
int metal_softmax_tensor(const void* src, void* dst, int num_rows, int dim_size);
int metal_logsumexp_tensor(const void* src, void* dst, int num_rows, int dim_size);
int metal_outer_tensor(const void* a, const void* b, void* dst, int M, int N);
int metal_dropout_tensor(const void* src, void* dst, int n, float p, int training, int seed);

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

int metal_layernorm_tensor(
    const void* src, void* dst, const void* weight, const void* bias,
    int num_rows, int d_model, float eps
);

int metal_rmsnorm_tensor(
    const void* src, void* dst, const void* weight,
    int num_rows, int d_model, float eps
);

int metal_groupnorm_tensor(
    const void* src, void* dst, const void* weight, const void* bias,
    int batch, int channels, int height, int width, int groups, float eps
);

// Scale: dst[i] = src[i] * (1/sqrt(dim)).
int metal_inv_sqrt_dim_scale(const float* src, float* dst, int n, int dim);

// Elementwise exp.
int metal_exp(const float* src, float* dst, int n);

// Elementwise log.
int metal_log(const float* src, float* dst, int n);

// Softmax over last dim. src/dst have num_rows*dim_size elements.
int metal_softmax(const float* src, float* dst, int num_rows, int dim_size);

// LogSumExp over last dim. src has num_rows*dim_size elements; dst has num_rows.
int metal_logsumexp(const float* src, float* dst, int num_rows, int dim_size);

// Dropout. If training is zero or p is zero, copies input to output.
int metal_dropout(const float* src, float* dst, int n, float p, int training, int seed);

// Layer norm. src/dst: num_rows*d_model; weight/bias: d_model.
int metal_layernorm(const float* src, float* dst,
                    const float* weight, const float* bias,
                    int num_rows, int d_model, float eps);

// RMS norm. src/dst: num_rows*d_model; weight: d_model.
int metal_rmsnorm(const float* src, float* dst,
                  const float* weight,
                  int num_rows, int d_model, float eps);

// Group norm over NCHW tensors. weight/bias length is channels.
int metal_groupnorm(
    const float* src, float* dst,
    const float* weight, const float* bias,
    int batch, int channels, int height, int width, int groups, float eps
);

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

// Training and benchmark primitives.
// metal_train_mse_loss writes a reduced scalar mean squared error to out[0].
// predictions and targets are length n; out must have length 1 and must not
// alias inputs. n==0 is a successful no-op with out[0]=0.
int metal_train_mse_loss(const float* predictions, const float* targets, float* out, int n);

// metal_train_cross_entropy_loss writes a reduced scalar softmax cross-entropy
// to out[0]. logits are raw pre-softmax scores and targets are probability or
// one-hot values in the same flat row-major class layout, both length n. out
// length is 1 and must not alias inputs.
int metal_train_cross_entropy_loss(const float* logits, const float* targets, float* out, int n);

// metal_train_mse_grad writes d(MSE)/d(predictions) into out[n]. predictions,
// targets, and out have length n. In-place use with out aliasing inputs is not
// supported.
int metal_train_mse_grad(const float* predictions, const float* targets, float* out, int n);

// metal_train_cross_entropy_grad writes softmax(logits)-targets into out[n].
// logits are raw scores and targets are probability/one-hot values matching the
// logits layout. In-place use with out aliasing inputs is not supported.
int metal_train_cross_entropy_grad(const float* logits, const float* targets, float* out, int n);
int metal_train_mse_loss_tensor(const void* predictions, const void* targets, void* out, int n);
int metal_train_mse_grad_tensor(const void* predictions, const void* targets, void* out, int n);
int metal_train_cross_entropy_loss_tensor(const void* logits, const void* targets, void* out, int n);
int metal_train_cross_entropy_grad_tensor(const void* logits, const void* targets, void* out, int n);
int metal_bench_accuracy_tensor(const void* predictions, const void* targets, void* out, int n);
int metal_bench_perplexity_tensor(const void* probabilities, const void* targets, void* out, int n);
int metal_bench_f1_tensor(const void* predictions, const void* targets, void* out, int n);

// metal_bench_accuracy writes one scalar to out[0]: 1 when prediction and target
// argmax indices match, otherwise 0. predictions and targets are length n score
// buffers.
int metal_bench_accuracy(const float* predictions, const float* targets, float* out, int n);

// metal_bench_f1_counts writes reduced binary confusion counts to out[0..2] as
// TP, FP, FN. predictions and targets are length n and thresholded at 0.5.
int metal_bench_f1_counts(const float* predictions, const float* targets, float* out, int n);

#ifdef __cplusplus
}
#endif

#endif /* METAL_KERNEL_MATH_H */
