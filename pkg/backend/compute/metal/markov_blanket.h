#ifndef METAL_MARKOV_BLANKET_H
#define METAL_MARKOV_BLANKET_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal pipelines for Markov blanket operations.
// metallib_path: absolute path to markov_blanket.metallib.
int metal_mb_init(const char* metallib_path);

// Partition joint state x[N] into [out_s|out_a|out_i|out_e].
// masks layout: [smask|amask|imask|emask] each length N (float32).
int metal_mb_partition(
    const float* x, const float* masks,
    float* out,
    int N, int Ns, int Na, int Ni, int Ne
);

// Internal flow: out[Ni] = W[Ni*Ns] @ x_sens[Ns] + bias[Ni]
int metal_mb_flow_internal(
    const float* x_sens, const float* W, const float* bias,
    float* out,
    int Ni, int Ns
);

// Active flow: out[Na] = W[Na*Ni] @ x_int[Ni] + bias[Na]
int metal_mb_flow_active(
    const float* x_int, const float* W, const float* bias,
    float* out,
    int Na, int Ni
);

// Mutual information (Gaussian approximation). X[T*N], Y[T*M] → out[1].
int metal_mb_mutual_information(
    const float* X, const float* Y,
    float* out,
    int T, int N, int M
);

#ifdef __cplusplus
}
#endif

#endif /* METAL_MARKOV_BLANKET_H */
