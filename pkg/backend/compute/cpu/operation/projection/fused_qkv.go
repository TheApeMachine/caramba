package projection

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
FusedQKV computes Q, K, V projections in a single matmul.

The state dict supplies:
  - OpShape: [M, DIn]
  - Inputs[0]: x, flattened [M * DIn]
  - Weight: pre-transposed [DIn * (DQ+DK+DV)]
  - Bias: optional [DQ+DK+DV]
  - DIn, DQ, DK, DV
*/
type FusedQKV struct{}

/*
NewFusedQKV instantiates a stateless FusedQKV operation.
*/
func NewFusedQKV(args ...int) *FusedQKV {
	return &FusedQKV{}
}

/*
NewFusedQKVWithSeed instantiates a stateless FusedQKV operation.
*/
func NewFusedQKVWithSeed(dIn, dQ, dK, dV int, seed int64) *FusedQKV {
	return &FusedQKV{}
}

/*
NewFusedQKVWithRNG instantiates a stateless FusedQKV operation.
*/
func NewFusedQKVWithRNG(dIn, dQ, dK, dV int, rng *rand.Rand) *FusedQKV {
	return &FusedQKV{}
}

/*
Forward computes output = x @ weight [+ bias].
*/
func (fqkv *FusedQKV) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("projection.fused_qkv"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("projection.fused_qkv: shape is required")
	}

	M := shape[0]
	K := stateDict.DIn

	if len(shape) > 1 && K == 0 {
		K = shape[len(shape)-1]
	}

	N := stateDict.DQ + stateDict.DK + stateDict.DV

	if M <= 0 {
		return nil, fmt.Errorf("projection.fused_qkv: M must be positive, got %d", M)
	}

	if K <= 0 {
		return nil, fmt.Errorf("projection.fused_qkv: d_in must be positive, got %d", K)
	}

	if N <= 0 {
		return nil, fmt.Errorf("projection.fused_qkv: DQ+DK+DV must be positive, got %d", N)
	}

	if M > len(stateDict.Inputs[0])/K {
		return nil, fmt.Errorf(
			"projection.fused_qkv: input length %d is insufficient for M=%d and K=%d",
			len(stateDict.Inputs[0]), M, K,
		)
	}

	if int64(K)*int64(N) < 0 || int64(K)*int64(N) > int64(math.MaxInt) {
		return nil, fmt.Errorf("projection.fused_qkv: K*N overflows int")
	}

	if len(stateDict.Weight) != K*N {
		return nil, fmt.Errorf(
			"projection.fused_qkv: weight length %d does not match K*N=%d",
			len(stateDict.Weight), K*N,
		)
	}

	if len(stateDict.Bias) != 0 && len(stateDict.Bias) != N {
		return nil, fmt.Errorf(
			"projection.fused_qkv: bias length %d does not match N=%d",
			len(stateDict.Bias), N,
		)
	}

	if int64(M)*int64(N) < 0 || int64(M)*int64(N) > int64(math.MaxInt) {
		return nil, fmt.Errorf("projection.fused_qkv: M*N overflows int")
	}

	stateDict.EnsureOperationOutLen(M * N)
	fusedQKVKernel(
		stateDict.Out, stateDict.Inputs[0], stateDict.Weight, stateDict.Bias,
		M, K, N,
	)

	return stateDict, nil
}
