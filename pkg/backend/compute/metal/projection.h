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
int metal_linear(const float* src, const float* weight, const float* bias,
                 float* dst, int M, int K, int N);

// Fused QKV: same as metal_linear but with outDim = DQ+DK+DV.
int metal_fused_qkv(const float* src, const float* weight, const float* bias,
                    float* dst, int M, int K, int N);

int metal_fused_qkv_tensor(
    const void* src,
    const void* weight,
    const void* bias,
    void*       dst,
    int         M,
    int         K,
    int         N);

// Tied embedding logit projection: dst[M*V] = src[M*D] @ weight[D*V]
int metal_tied_embedding(const float* src, const float* weight,
                         float* dst, int M, int D, int V);

#ifdef __cplusplus
}
#endif

#endif /* METAL_PROJECTION_H */
