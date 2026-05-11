#ifndef XLA_MARKOV_BLANKET_H
#define XLA_MARKOV_BLANKET_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize XLA PJRT client for Markov blanket operations.
int xla_mb_init(const char* platform);

// Partition joint state x[N] into [s|a|i|e] partitions.
// masks[4*N]: [smask|amask|imask|emask].
int xla_mb_partition(
    const double* x, const double* masks,
    double* out,
    int N, int Ns, int Na, int Ni, int Ne
);

// Internal flow: out[Ni] = W[Ni*Ns] @ x_sens[Ns] + bias[Ni]
int xla_mb_flow_internal(
    const double* x_sens, const double* W, const double* bias,
    double* out,
    int Ni, int Ns
);

// Active flow: out[Na] = W[Na*Ni] @ x_int[Ni] + bias[Na]
int xla_mb_flow_active(
    const double* x_int, const double* W, const double* bias,
    double* out,
    int Na, int Ni
);

// Mutual information (Gaussian). X[T*N], Y[T*M] → out[1] scalar.
int xla_mb_mutual_information(
    const double* X, const double* Y,
    double* out,
    int T, int N, int M
);

void xla_mb_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* XLA_MARKOV_BLANKET_H */
