#ifndef CUDA_MARKOV_BLANKET_H
#define CUDA_MARKOV_BLANKET_H

#ifdef __cplusplus
extern "C" {
#endif

// Partition joint state x[N] into sensory/active/internal/external sub-vectors.
// masks[0..N-1] sensory, masks[N..2N-1] active, masks[2N..3N-1] internal, masks[3N..4N-1] external.
// out[0..Ns+Na+Ni+Ne-1]: concatenation of the four partitions.
int cuda_mb_partition(
    const double* x, const double* masks,
    double* out,
    int N, int Ns, int Na, int Ni, int Ne
);

// Internal flow: out[Ni] = W[Ni*Ns] @ x_sens[Ns] + bias[Ni]
int cuda_mb_flow_internal(
    const double* x_sens, const double* W, const double* bias,
    double* out,
    int Ni, int Ns
);

// Active flow: out[Na] = W[Na*Ni] @ x_int[Ni] + bias[Na]
int cuda_mb_flow_active(
    const double* x_int, const double* W, const double* bias,
    double* out,
    int Na, int Ni
);

// Mutual information (Gaussian approximation): returns scalar MI into out[1].
// X [T*N], Y [T*M].
int cuda_mb_mutual_information(
    const double* X, const double* Y,
    double* out,
    int T, int N, int M
);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_MARKOV_BLANKET_H */
