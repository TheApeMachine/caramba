//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "positional.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// MetalPositional dispatches RoPE and ALiBi kernels to the Metal GPU.
type MetalPositional struct {
	metallib string
}

// NewPositional creates and initializes a MetalPositional.
// metallib must be the absolute path to positional.metallib.
func NewPositional(metallib string) (*MetalPositional, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))
	if rc := C.metal_positional_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_positional_init failed (rc=%d)", rc)
	}
	return &MetalPositional{metallib: metallib}, nil
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

// buildTablesFP32 returns interleaved cos/sin tables as []float32.
func buildTablesFP32(seqLen, headDim int, base float64) ([]float32, []float32) {
	numPairs := headDim / 2
	n := seqLen * numPairs
	cosT := make([]float32, n)
	sinT := make([]float32, n)
	for t := 0; t < seqLen; t++ {
		for i := 0; i < numPairs; i++ {
			theta := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
			angle := float64(t) * theta
			cosT[t*numPairs+i] = float32(math.Cos(angle))
			sinT[t*numPairs+i] = float32(math.Sin(angle))
		}
	}
	return cosT, sinT
}

// RoPEForward applies rotary position embeddings on the Metal GPU.
// shape=[batch, num_heads, seq_len, head_dim]; data[0]=input tensor.
func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	if base == 0 {
		base = 10000.0
	}

	x := data[0]
	src32 := toFloat32(x)
	dst32 := make([]float32, len(x))

	cosT, sinT := buildTablesFP32(seqLen, headDim, base)

	totalHeads := batch * numHeads
	rc := C.metal_rope(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		(*C.float)(unsafe.Pointer(&cosT[0])),
		(*C.float)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(totalHeads),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_rope failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// Forward dispatches RoPE with the universal signature.
// shape=[batch, num_heads, seq_len, head_dim]; uses default base=10000.
func (m *MetalPositional) Forward(shape []int, data ...[]float64) []float64 {
	out, _ := m.RoPEForward(10000.0, shape, data...)
	return out
}

// ---------------------------------------------------------------------------
// ALiBi
// ---------------------------------------------------------------------------

// buildSlopesFP32 computes ALiBi head slopes as float32.
func buildSlopesFP32(numHeads int) []float32 {
	n := 1
	for n < numHeads {
		n <<= 1
	}
	pow2 := func(k int) []float32 {
		s := make([]float32, k)
		step := 8.0 / float64(k)
		for i := range s {
			s[i] = float32(math.Pow(2, -step*float64(i+1)))
		}
		return s
	}
	if n == numHeads {
		return pow2(n)
	}
	half := pow2(n / 2)
	full := pow2(n)
	slopes := make([]float32, numHeads)
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

// ALiBiForward computes the ALiBi bias tensor on the Metal GPU.
// shape=[num_heads, seq_len_q, seq_len_k].
func (m *MetalPositional) ALiBiForward(shape []int) ([]float64, error) {
	numHeads := shape[0]
	seqLenQ := shape[1]
	seqLenK := shape[2]

	slopes := buildSlopesFP32(numHeads)
	dst32 := make([]float32, numHeads*seqLenQ*seqLenK)

	rc := C.metal_alibi(
		(*C.float)(unsafe.Pointer(&dst32[0])),
		(*C.float)(unsafe.Pointer(&slopes[0])),
		C.int(numHeads),
		C.int(seqLenQ),
		C.int(seqLenK),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_alibi failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}
