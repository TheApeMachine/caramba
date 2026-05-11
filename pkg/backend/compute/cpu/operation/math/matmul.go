package math

import (
	"fmt"
	"math"
)

/*
Matmul performs matrix multiplication A [M*K] x B [K*N] -> C [M*N].
shape = [M, K, N].
*/
type Matmul struct{}

func NewMatmul() *Matmul { return &Matmul{} }

func (op *Matmul) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 3 {
		panic(fmt.Errorf("math: Matmul.Forward: len(shape)=%d, need >= 3", len(shape)).Error())
	}

	M, K, N := shape[0], shape[1], shape[2]

	if len(data) < 2 {
		panic(fmt.Errorf("math: Matmul.Forward: len(data)=%d, need >= 2", len(data)).Error())
	}

	mk := int64(M) * int64(K)
	kn := int64(K) * int64(N)
	mn := int64(M) * int64(N)

	if mk < 0 || mk > int64(math.MaxInt) {
		panic(fmt.Errorf("math: Matmul.Forward: M*K overflows int (M=%d K=%d)", M, K).Error())
	}

	if kn < 0 || kn > int64(math.MaxInt) {
		panic(fmt.Errorf("math: Matmul.Forward: K*N overflows int (K=%d N=%d)", K, N).Error())
	}

	if mn < 0 || mn > int64(math.MaxInt) {
		panic(fmt.Errorf("math: Matmul.Forward: M*N overflows int (M=%d N=%d)", M, N).Error())
	}

	if len(data[0]) != int(mk) {
		panic(fmt.Errorf(
			"math: Matmul.Forward: len(data[0])=%d, need M*K=%d (M=%d K=%d)",
			len(data[0]), int(mk), M, K,
		).Error())
	}

	if len(data[1]) != int(kn) {
		panic(fmt.Errorf(
			"math: Matmul.Forward: len(data[1])=%d, need K*N=%d (K=%d N=%d)",
			len(data[1]), int(kn), K, N,
		).Error())
	}

	out := make([]float64, M*N)
	applyMatMul(out, data[0], data[1], M, K, N)

	return out
}
