#ifndef CARAMBA_XLA_TENSOR_H
#define CARAMBA_XLA_TENSOR_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque PJRT-backed resident tensor; internal layout is implementation-defined. */
typedef struct XLA_Tensor XLA_Tensor;

/**
 * @brief Initialise PJRT client/device state for the tensor backend.
 * @param platform NUL-terminated name (e.g. "cpu", "gpu"); must not be NULL.
 * @return 0 on success; non-zero (typically -1) on failure (plugin load, client create).
 * @note Not guaranteed thread-safe; serialise with other tensor APIs on the same process.
 * @note Behaviour after failure: safe to retry init; no tensor handles are valid until success.
 */
int xla_tensor_init(const char* platform);

/**
 * @brief Tear down PJRT state for this backend (executables, client).
 * @note Idempotent after successful init; safe if init never succeeded (no-op cleanup).
 * @note Calling tensor ops concurrently with shutdown is undefined; serialize externally.
 */
void xla_tensor_shutdown(void);

/**
 * @brief Allocate a device tensor and upload host float64 values.
 * @param src Host row-major data; may be NULL only when element count is zero.
 * @param dims Length @p rank; must be non-NULL when rank > 0.
 * @param rank Number of dimensions (non-negative).
 * @param out Must be non-NULL. On success *out is a new tensor (caller must xla_tensor_free).
 *            On failure *out is NULL.
 * @return 0 on success; non-zero (typically -1) on invalid arguments or PJRT failure.
 * @note Ownership: *out belongs to caller until xla_tensor_free.
 */
int xla_tensor_upload_f64(
	const double* src, const int64_t* dims, int rank, XLA_Tensor** out
);

/**
 * @brief Copy device tensor contents to host memory (blocking).
 * @param tensor Non-NULL tensor from upload or unary/binary outputs.
 * @param dst Non-NULL buffer with space for @p n_elements doubles.
 * @param n_elements Number of double elements in @p dst; must equal the tensor's scalar count.
 * @return 0 on success; non-zero (typically -1) if tensor/dst invalid or count mismatch / PJRT error.
 */
int xla_tensor_download_f64(
	const XLA_Tensor* tensor, double* dst, int64_t n_elements
);

/**
 * @brief Release a tensor allocated by upload or kernel outputs.
 * @param tensor May be NULL (no-op). Non-NULL tensor must not be used after call.
 * @return 0 on success; non-zero if PJRT buffer destroy fails.
 */
int xla_tensor_free(XLA_Tensor* tensor);

/** Unary kernels: @p output non-NULL; *output receives new tensor on success. Return 0 or non-zero. */
int xla_tensor_relu(const XLA_Tensor* input, XLA_Tensor** output);

/**
 * @brief LeakyReLU with coefficient @p alpha (kernel-defined formula).
 * @param output Last parameter by convention (alpha before output pointer).
 */
int xla_tensor_leaky_relu(const XLA_Tensor* input, double alpha, XLA_Tensor** output);

int xla_tensor_gelu(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_tanh(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_sigmoid(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_swish(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_selu(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_swiglu(const XLA_Tensor* input, XLA_Tensor** output);

int xla_tensor_add(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output);
int xla_tensor_mul(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output);
int xla_tensor_matmul(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output);

int xla_tensor_reshape(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	XLA_Tensor** output
);

int xla_tensor_transpose(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	int dim0,
	int dim1,
	XLA_Tensor** output
);

int xla_tensor_concat(
	const XLA_Tensor* left,
	const XLA_Tensor* right,
	const int64_t* output_dims,
	int output_rank,
	XLA_Tensor** output
);

int xla_tensor_split(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	int outer,
	int dim_size,
	int split_size,
	int inner,
	XLA_Tensor** output
);

int xla_tensor_upsample_nearest2d(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	int batch,
	int channels,
	int height,
	int width,
	int scale_h,
	int scale_w,
	XLA_Tensor** output
);

int xla_tensor_view_as_heads(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	int batch,
	int tokens,
	int heads,
	int head_dim,
	XLA_Tensor** output
);

int xla_tensor_merge_heads(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	int batch,
	int heads,
	int tokens,
	int head_dim,
	XLA_Tensor** output
);

int xla_tensor_last_token(
	const XLA_Tensor* input,
	const int64_t* output_dims,
	int output_rank,
	int outer,
	int seq_len,
	int feature,
	XLA_Tensor** output
);

/**
 * @brief Fused matmul(left, right) + bias; optionally GELU on each output element.
 * @param apply_gelu If true, apply GELU to the fused result.
 */
int xla_tensor_matmul_add(
	const XLA_Tensor* left,
	const XLA_Tensor* right,
	const XLA_Tensor* bias,
	XLA_Tensor** output,
	bool apply_gelu
);

#ifdef __cplusplus
}
#endif

#endif /* CARAMBA_XLA_TENSOR_H */
