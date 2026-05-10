//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "positional.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// CUDAPositionalOps dispatches RoPE and ALiBi kernels to the CUDA GPU.
type CUDAPositionalOps struct{}

// NewPositional returns a CUDAPositionalOps.
func NewPositional() *CUDAPositionalOps {
	return &CUDAPositionalOps{}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func buildCossinTables(seqLen, headDim int, base float64) ([]float64, []float64) {
	numPairs := headDim / 2
	n := seqLen * numPairs
	cosT := make([]float64, n)
	sinT := make([]float64, n)
	for t := 0; t < seqLen; t++ {
		for i := 0; i < numPairs; i++ {
			theta := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
			angle := float64(t) * theta
			cosT[t*numPairs+i] = math.Cos(angle)
			sinT[t*numPairs+i] = math.Sin(angle)
		}
	}
	return cosT, sinT
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
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]
	if base == 0 {
		base = 10000.0
	}

	x := data[0]
	dst := make([]float64, len(x))
	cosT, sinT := buildCossinTables(seqLen, headDim, base)
	totalHeads := batch * numHeads

	rc := C.cuda_rope(
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&cosT[0])),
		(*C.double)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(totalHeads),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_rope failed (rc=%d)", rc)
	}
	return dst, nil
}

// Forward dispatches RoPE with the universal signature.
func (c *CUDAPositionalOps) Forward(shape []int, data ...[]float64) []float64 {
	out, _ := c.RoPEForward(10000.0, shape, data...)
	return out
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
