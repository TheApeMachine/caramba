#ifndef CARAMBA_XLA_TENSOR_H
#define CARAMBA_XLA_TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XLA_Tensor XLA_Tensor;

int xla_tensor_init(const char* platform);
void xla_tensor_shutdown(void);

XLA_Tensor* xla_tensor_upload_f64(const double* src, const int64_t* dims, int rank);
int xla_tensor_download_f64(const XLA_Tensor* tensor, double* dst, int n);
int xla_tensor_free(XLA_Tensor* tensor);

int xla_tensor_relu(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_leaky_relu(const XLA_Tensor* input, XLA_Tensor** output, double alpha);
int xla_tensor_gelu(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_tanh(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_sigmoid(const XLA_Tensor* input, XLA_Tensor** output);
int xla_tensor_swiglu(const XLA_Tensor* input, XLA_Tensor** output);

int xla_tensor_add(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output);
int xla_tensor_mul(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output);
int xla_tensor_matmul(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output);
int xla_tensor_matmul_add(
	const XLA_Tensor* left,
	const XLA_Tensor* right,
	const XLA_Tensor* bias,
	XLA_Tensor** output,
	int apply_gelu
);

#ifdef __cplusplus
}
#endif

#endif /* CARAMBA_XLA_TENSOR_H */
