#ifndef CUDA_PROJECTION_H
#define CUDA_PROJECTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Linear projection: dst[M*N] = src[M*K] @ weight[N*K]^T  (+bias if non-NULL)
// weight is stored [N*K] row-major (each row = output-neuron weight vector).
// Returns 0 on success, -1 on CUDA error.
int cuda_linear(const double* src, const double* weight, const double* bias,
                double* dst, int M, int K, int N, int has_bias);

// Fused QKV projection: identical layout, outDim = DQ+DK+DV.
int cuda_fused_qkv(const double* src, const double* weight, const double* bias,
                   double* dst, int M, int K, int N, int has_bias);

// Tied embedding logit projection: dst[M*V] = src[M*D] @ weight[V*D]^T
int cuda_tied_embedding(const double* src, const double* weight,
                        double* dst, int M, int D, int V);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PROJECTION_H */
