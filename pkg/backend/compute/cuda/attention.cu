#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include "attention.h"

// ---------------------------------------------------------------------------
// Each kernel processes one (batch, head) pair == one block.
// Dynamic shared memory holds the score row (seq_len doubles).
// Block dimension: min(seq_len, 1024).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Warp-level parallel reduction helpers
// ---------------------------------------------------------------------------

__device__ static inline double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ static inline double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-wide max using shared memory + warp shuffles.
__device__ static double block_reduce_max(double val, double* smem_scratch) {
    int lane  = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem_scratch[warpId] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? smem_scratch[threadIdx.x] : -DBL_MAX;
    if (warpId == 0) val = warp_reduce_max(val);
    __syncthreads();
    if (threadIdx.x == 0) smem_scratch[0] = val;
    __syncthreads();
    return smem_scratch[0];
}

__device__ static double block_reduce_sum(double val, double* smem_scratch) {
    int lane  = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_scratch[warpId] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? smem_scratch[threadIdx.x] : 0.0;
    if (warpId == 0) val = warp_reduce_sum(val);
    __syncthreads();
    if (threadIdx.x == 0) smem_scratch[0] = val;
    __syncthreads();
    return smem_scratch[0];
}

// ---------------------------------------------------------------------------
// SDPA kernel
// grid = batch * num_heads blocks; each block handles one head.
// extern __shared__: seq_len doubles (scores) + 32 doubles (reduction scratch).
// ---------------------------------------------------------------------------

__global__ void sdpa_kernel(
    const double* q, const double* k, const double* v,
    double* out,
    int seq_len, int head_dim)
{
    extern __shared__ double scores[];
    double* scratch = scores + seq_len; // 32 doubles for warp reductions

    int head_offset = blockIdx.x * seq_len * head_dim;
    int i = threadIdx.x; // query position handled by this thread (may be >= seq_len)

    double scale = 1.0 / sqrt((double)head_dim);

    // Step 1: each thread computes scores for its query position (if i < seq_len)
    if (i < seq_len) {
        const double* q_row = q + head_offset + i * head_dim;
        for (int j = 0; j < seq_len; j++) {
            const double* k_row = k + head_offset + j * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
            scores[i * seq_len + j] = dot * scale; // store in row-major in smem
        }
    }
    __syncthreads();

    // Step 2: softmax per row i (each thread owns row i)
    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double max_val = row[0];
        for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);
        double sum = 0.0;
        for (int j = 0; j < seq_len; j++) { row[j] = exp(row[j] - max_val); sum += row[j]; }
        for (int j = 0; j < seq_len; j++) row[j] /= sum;
    }
    __syncthreads();

    // Step 3: weighted sum of V
    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double* out_row = out + head_offset + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (int j = 0; j < seq_len; j++) acc += row[j] * v[head_offset + j * head_dim + d];
            out_row[d] = acc;
        }
    }
    (void)scratch; // suppress unused warning; kept for potential future use
}

// ---------------------------------------------------------------------------
// MQA kernel — K/V broadcast from head 0 (kv_head_offset = 0 within each batch item)
// ---------------------------------------------------------------------------

__global__ void mqa_kernel(
    const double* q, const double* k, const double* v,
    double* out,
    int seq_len, int head_dim,
    int num_heads)
{
    extern __shared__ double scores[];

    // blockIdx.x is the flattened (batch_idx * num_heads + head_idx)
    int batch_idx = blockIdx.x / num_heads;
    // head_idx within batch (unused for kv offset — KV always uses head 0 of the batch)
    int q_head_offset  = blockIdx.x * seq_len * head_dim;
    int kv_head_offset = batch_idx  * seq_len * head_dim; // 1 KV head per batch item

    int i = threadIdx.x;
    double scale = 1.0 / sqrt((double)head_dim);

    if (i < seq_len) {
        const double* q_row = q + q_head_offset + i * head_dim;
        for (int j = 0; j < seq_len; j++) {
            const double* k_row = k + kv_head_offset + j * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
            scores[i * seq_len + j] = dot * scale;
        }
    }
    __syncthreads();

    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double max_val = row[0];
        for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);
        double sum = 0.0;
        for (int j = 0; j < seq_len; j++) { row[j] = exp(row[j] - max_val); sum += row[j]; }
        for (int j = 0; j < seq_len; j++) row[j] /= sum;
    }
    __syncthreads();

    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double* out_row = out + q_head_offset + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (int j = 0; j < seq_len; j++) acc += row[j] * v[kv_head_offset + j * head_dim + d];
            out_row[d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// GQA kernel
// ---------------------------------------------------------------------------

__global__ void gqa_kernel(
    const double* q, const double* k, const double* v,
    double* out,
    int seq_len, int head_dim,
    int num_heads, int num_kv_heads)
{
    extern __shared__ double scores[];

    int group_size    = num_heads / num_kv_heads;
    int batch_idx     = blockIdx.x / num_heads;
    int local_head    = blockIdx.x % num_heads;
    int kv_local_head = local_head / group_size;

    int q_head_offset  = blockIdx.x  * seq_len * head_dim;
    int kv_head_offset = (batch_idx * num_kv_heads + kv_local_head) * seq_len * head_dim;

    int i = threadIdx.x;
    double scale = 1.0 / sqrt((double)head_dim);

    if (i < seq_len) {
        const double* q_row = q + q_head_offset + i * head_dim;
        for (int j = 0; j < seq_len; j++) {
            const double* k_row = k + kv_head_offset + j * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
            scores[i * seq_len + j] = dot * scale;
        }
    }
    __syncthreads();

    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double max_val = row[0];
        for (int j = 1; j < seq_len; j++) max_val = max(max_val, row[j]);
        double sum = 0.0;
        for (int j = 0; j < seq_len; j++) { row[j] = exp(row[j] - max_val); sum += row[j]; }
        for (int j = 0; j < seq_len; j++) row[j] /= sum;
    }
    __syncthreads();

    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double* out_row = out + q_head_offset + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (int j = 0; j < seq_len; j++) acc += row[j] * v[kv_head_offset + j * head_dim + d];
            out_row[d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding Window kernel
// ---------------------------------------------------------------------------

__global__ void sliding_window_kernel(
    const double* q, const double* k, const double* v,
    double* out,
    int seq_len, int head_dim, int window)
{
    extern __shared__ double scores[];

    int head_offset = blockIdx.x * seq_len * head_dim;
    int i = threadIdx.x;
    double scale = 1.0 / sqrt((double)head_dim);

    if (i < seq_len) {
        const double* q_row = q + head_offset + i * head_dim;
        for (int j = 0; j < seq_len; j++) {
            double score;
            if (j < i - window || j > i) {
                score = -DBL_MAX;
            } else {
                const double* k_row = k + head_offset + j * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; d++) dot += q_row[d] * k_row[d];
                score = dot * scale;
            }
            scores[i * seq_len + j] = score;
        }
    }
    __syncthreads();

    if (i < seq_len) {
        double* row = scores + i * seq_len;
        // Find max among non-masked entries
        double max_val = -DBL_MAX;
        for (int j = 0; j < seq_len; j++)
            if (row[j] != -DBL_MAX) max_val = max(max_val, row[j]);
        double sum = 0.0;
        for (int j = 0; j < seq_len; j++) {
            if (row[j] == -DBL_MAX) { row[j] = 0.0; }
            else { row[j] = exp(row[j] - max_val); sum += row[j]; }
        }
        if (sum > 0.0) for (int j = 0; j < seq_len; j++) row[j] /= sum;
    }
    __syncthreads();

    if (i < seq_len) {
        double* row = scores + i * seq_len;
        double* out_row = out + head_offset + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            double acc = 0.0;
            for (int j = 0; j < seq_len; j++) acc += row[j] * v[head_offset + j * head_dim + d];
            out_row[d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// C linkage wrappers
// ---------------------------------------------------------------------------

extern "C" {

int cuda_sdpa(const double* q, const double* k, const double* v, double* out,
              int batch, int num_heads, int seq_len, int head_dim)
{
    int total_heads = batch * num_heads;
    size_t qkv_bytes = (size_t)total_heads * seq_len * head_dim * sizeof(double);
    size_t out_bytes = qkv_bytes;
    size_t smem_bytes = (size_t)seq_len * seq_len * sizeof(double);

    double *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_out = NULL;
    if (cudaMalloc(&d_q,   qkv_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_k,   qkv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_v,   qkv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_out, out_bytes)  != cudaSuccess) goto fail;

    if (cudaMemcpy(d_q, q, qkv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_k, k, qkv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_v, v, qkv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    {
        int block = seq_len < 1024 ? seq_len : 1024;
        sdpa_kernel<<<total_heads, block, smem_bytes>>>(d_q, d_k, d_v, d_out, seq_len, head_dim);
    }
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return -1;
}

int cuda_mqa(const double* q, const double* k, const double* v, double* out,
             int batch, int num_heads, int seq_len, int head_dim)
{
    int total_q_heads  = batch * num_heads;
    int total_kv_heads = batch * 1;
    size_t q_bytes  = (size_t)total_q_heads  * seq_len * head_dim * sizeof(double);
    size_t kv_bytes = (size_t)total_kv_heads * seq_len * head_dim * sizeof(double);
    size_t smem_bytes = (size_t)seq_len * seq_len * sizeof(double);

    double *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_out = NULL;
    if (cudaMalloc(&d_q,   q_bytes)  != cudaSuccess) return -1;
    if (cudaMalloc(&d_k,   kv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_v,   kv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_out, q_bytes)  != cudaSuccess) goto fail;

    if (cudaMemcpy(d_q, q, q_bytes,  cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_k, k, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_v, v, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    {
        int block = seq_len < 1024 ? seq_len : 1024;
        mqa_kernel<<<total_q_heads, block, smem_bytes>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, num_heads);
    }
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, d_out, q_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return -1;
}

int cuda_gqa(const double* q, const double* k, const double* v, double* out,
             int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim)
{
    int total_q_heads  = batch * num_heads;
    int total_kv_heads = batch * num_kv_heads;
    size_t q_bytes  = (size_t)total_q_heads  * seq_len * head_dim * sizeof(double);
    size_t kv_bytes = (size_t)total_kv_heads * seq_len * head_dim * sizeof(double);
    size_t smem_bytes = (size_t)seq_len * seq_len * sizeof(double);

    double *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_out = NULL;
    if (cudaMalloc(&d_q,   q_bytes)  != cudaSuccess) return -1;
    if (cudaMalloc(&d_k,   kv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_v,   kv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_out, q_bytes)  != cudaSuccess) goto fail;

    if (cudaMemcpy(d_q, q, q_bytes,  cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_k, k, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_v, v, kv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    {
        int block = seq_len < 1024 ? seq_len : 1024;
        gqa_kernel<<<total_q_heads, block, smem_bytes>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, num_heads, num_kv_heads);
    }
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, d_out, q_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return -1;
}

int cuda_sliding_window(const double* q, const double* k, const double* v, double* out,
                        int batch, int num_heads, int seq_len, int head_dim, int window)
{
    int total_heads = batch * num_heads;
    size_t qkv_bytes  = (size_t)total_heads * seq_len * head_dim * sizeof(double);
    size_t smem_bytes = (size_t)seq_len * seq_len * sizeof(double);

    double *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_out = NULL;
    if (cudaMalloc(&d_q,   qkv_bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_k,   qkv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_v,   qkv_bytes) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_out, qkv_bytes) != cudaSuccess) goto fail;

    if (cudaMemcpy(d_q, q, qkv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_k, k, qkv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;
    if (cudaMemcpy(d_v, v, qkv_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

    {
        int block = seq_len < 1024 ? seq_len : 1024;
        sliding_window_kernel<<<total_heads, block, smem_bytes>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, window);
    }
    if (cudaGetLastError() != cudaSuccess) goto fail;
    if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    if (cudaMemcpy(out, d_out, qkv_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto fail;

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return 0;
fail:
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    return -1;
}

} // extern "C"
