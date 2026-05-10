#ifndef METAL_PROJECTION_H
#define METAL_PROJECTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and projection compute pipelines.
// metallib_path: absolute path to the compiled projection.metallib.
// Returns 0 on success, -1 on failure.
int metal_projection_init(const char* metallib_path);

// Linear projection: dst[M*N] = src[M*K] @ weight[K*N]  (+bias if non-NULL)
// weight is stored [K*N] column-major (i.e. weight^T in row-major) so that
// each column of weight^T is a contiguous output-neuron row.
// In practice we pass weight as [N*K] row-major and transpose on the GPU.
int metal_linear(const float* src, const float* weight, const float* bias,
                 float* dst, int M, int K, int N);

// Fused QKV: same as metal_linear but with outDim = DQ+DK+DV.
int metal_fused_qkv(const float* src, const float* weight, const float* bias,
                    float* dst, int M, int K, int N);

// Tied embedding logit projection: dst[M*V] = src[M*D] @ weight[V*D]^T
int metal_tied_embedding(const float* src, const float* weight,
                         float* dst, int M, int D, int V);

#ifdef __cplusplus
}
#endif

#endif /* METAL_PROJECTION_H */
