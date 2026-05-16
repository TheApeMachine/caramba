//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "positional.h"
import "C"

import (
	"fmt"
	"math"
	"strings"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
)

// CUDAPositionalOps dispatches RoPE and ALiBi kernels to the CUDA GPU.
type CUDAPositionalOps struct{}

const (
	cudaRoPEModeAdjacent = 0
	cudaRoPEModeHalf     = 1
)

// NewPositional returns a CUDAPositionalOps.
func NewPositional() *CUDAPositionalOps {
	return &CUDAPositionalOps{}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func buildCossinTables(
	seqLen int,
	headDim int,
	config rotary.Config,
	positionStart int,
) ([]float64, []float64, error) {
	numPairs := headDim / 2
	n := seqLen * numPairs
	cosT := make([]float64, n)
	sinT := make([]float64, n)
	frequencies, err := config.InverseFrequencies(headDim)

	if err != nil {
		return nil, nil, err
	}

	for t := 0; t < seqLen; t++ {
		for i := 0; i < numPairs; i++ {
			theta := frequencies[i]
			angle := float64(positionStart+t) * theta
			cosT[t*numPairs+i] = math.Cos(angle)
			sinT[t*numPairs+i] = math.Sin(angle)
		}
	}

	return cosT, sinT, nil
}

func buildSlopesFloat64(numHeads int) []float64 {
	n := 1
	for n < numHeads {
		n <<= 1
	}
	pow2 := func(k int) []float64 {
		s := make([]float64, k)
		step := 8.0 / float64(k)
		for i := range s {
			s[i] = math.Pow(2, -step*float64(i+1))
		}
		return s
	}
	if n == numHeads {
		return pow2(n)
	}
	half := pow2(n / 2)
	full := pow2(n)
	slopes := make([]float64, numHeads)
	hi, fi := 0, 0
	for i := 0; i < numHeads; i++ {
		if i%2 == 0 && hi < len(half) {
			slopes[i] = half[hi]
			hi++
		} else if fi < len(full) {
			slopes[i] = full[fi]
			fi++
		}
	}
	return slopes
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

// RoPEForward applies rotary position embeddings.
// shape=[batch, num_heads, seq_len, head_dim]; data[0]=input tensor.
func (c *CUDAPositionalOps) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	return c.RoPEForwardAt(base, 0, shape, data...)
}

func (c *CUDAPositionalOps) RoPEForwardAt(
	base float64,
	positionStart int,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return c.RoPEForwardAtMode(base, positionStart, "", shape, data...)
}

func (c *CUDAPositionalOps) RoPEForwardAtMode(
	base float64,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return c.RoPEForwardAtModeConfig(
		rotary.Config{Base: base},
		positionStart,
		mode,
		shape,
		data...,
	)
}

func (c *CUDAPositionalOps) RoPEForwardAtModeConfig(
	config rotary.Config,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cuda_rope: input[0] is required")
	}

	if len(shape) != 4 {
		return nil, fmt.Errorf("cuda_rope: expected rank 4 shape, got %d", len(shape))
	}

	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	if batch <= 0 || numHeads <= 0 || seqLen <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("cuda_rope: all shape dimensions must be positive")
	}

	if headDim%2 != 0 {
		return nil, fmt.Errorf("cuda_rope: expected even head_dim, got %d", headDim)
	}

	x := data[0]

	if len(x) != batch*numHeads*seqLen*headDim {
		return nil, fmt.Errorf("cuda_rope: input length mismatch")
	}

	dst := make([]float64, len(x))
	cosT, sinT, err := buildCossinTables(seqLen, headDim, config, positionStart)

	if err != nil {
		return nil, err
	}

	modeCode, err := cudaRoPEMode(mode)

	if err != nil {
		return nil, err
	}

	totalHeads := batch * numHeads

	rc := C.cuda_rope(
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&cosT[0])),
		(*C.double)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(modeCode),
		C.int(totalHeads),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_rope failed (rc=%d)", rc)
	}
	return dst, nil
}

func cudaRoPEMode(mode string) (int, error) {
	switch strings.ToLower(mode) {
	case "", "adjacent":
		return cudaRoPEModeAdjacent, nil
	case "half":
		return cudaRoPEModeHalf, nil
	default:
		return 0, fmt.Errorf("cuda_rope: unsupported mode %q", mode)
	}
}

// Forward dispatches RoPE with the universal signature.
func (c *CUDAPositionalOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cuda positional: Forward requires data[0]")
	}

	return c.RoPEForward(10000.0, shape, data...)
}

// ---------------------------------------------------------------------------
// ALiBi
// ---------------------------------------------------------------------------

// ALiBiForward computes the ALiBi bias tensor.
// shape=[num_heads, seq_len_q, seq_len_k].
func (c *CUDAPositionalOps) ALiBiForward(shape []int, causal bool) ([]float64, error) {
	numHeads := shape[0]
	seqLenQ := shape[1]
	seqLenK := shape[2]

	slopes := buildSlopesFloat64(numHeads)
	dst := make([]float64, numHeads*seqLenQ*seqLenK)

	causalInt := 0
	if causal {
		causalInt = 1
	}

	rc := C.cuda_alibi(
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&slopes[0])),
		C.int(numHeads),
		C.int(seqLenQ),
		C.int(seqLenK),
		C.int(causalInt),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_alibi failed (rc=%d)", rc)
	}
	return dst, nil
}
