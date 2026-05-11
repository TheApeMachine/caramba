//go:build cgo && xla

package xla

// XLA positional backend via the PJRT C API.
//
// Build requirements:
//   - XLA headers on the include path (set XLA_INCLUDE via CGO_CPPFLAGS)
//   - PJRT plugin shared library for your platform on LD_LIBRARY_PATH
//   - Compile positional_xla.cc alongside this package (CGo picks it up)
//
// Example build:
//   CGO_CPPFLAGS="-I/path/to/xla" \
//   CGO_LDFLAGS="-ldl -lstdc++" \
//   go build -tags "cgo xla" ./backend/compute/xla/

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl -lstdc++
// #include <stdlib.h>
// #include "positional.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// XLAPositionalOps dispatches RoPE and ALiBi to the XLA runtime via PJRT.
type XLAPositionalOps struct {
	platform string
}

// NewPositional initialises the PJRT client for the given platform.
func NewPositional(platform string) (*XLAPositionalOps, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))
	if rc := C.xla_positional_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_positional_init failed for platform %q", platform)
	}
	return &XLAPositionalOps{platform: platform}, nil
}

// Shutdown releases all PJRT resources.
func (x *XLAPositionalOps) Shutdown() {
	C.xla_positional_shutdown()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func xlaCosSinTables(seqLen, headDim int, base float64) ([]float64, []float64) {
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

func xlaSlopes(numHeads int) []float64 {
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

// ensureCompiledRoPE compiles or reuses existing executables for the given dims.
func (x *XLAPositionalOps) ensureCompiledRoPE(totalHeads, seqLen, headDim int) error {
	rc := C.xla_compile_positional(
		C.int(totalHeads), C.int(seqLen), C.int(headDim),
		C.int(1), C.int(1), C.int(1),
	)
	if rc != 0 {
		return fmt.Errorf("xla_compile_positional (rope) failed")
	}
	return nil
}

func (x *XLAPositionalOps) ensureCompiledALiBi(numHeads, seqLenQ, seqLenK int) error {
	rc := C.xla_compile_positional(
		C.int(1), C.int(1), C.int(2),
		C.int(numHeads), C.int(seqLenQ), C.int(seqLenK),
	)
	if rc != 0 {
		return fmt.Errorf("xla_compile_positional (alibi) failed")
	}
	return nil
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

// RoPEForward applies rotary position embeddings.
// shape=[batch, num_heads, seq_len, head_dim]; data[0]=input tensor.
func (x *XLAPositionalOps) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]
	if base == 0 {
		base = 10000.0
	}
	totalHeads := batch * numHeads

	if err := x.ensureCompiledRoPE(totalHeads, seqLen, headDim); err != nil {
		return nil, err
	}

	input := data[0]
	dst := make([]float64, len(input))
	cosT, sinT := xlaCosSinTables(seqLen, headDim, base)

	rc := C.xla_rope(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&cosT[0])),
		(*C.double)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(totalHeads),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_rope failed")
	}
	return dst, nil
}

// Forward dispatches RoPE with the universal signature.
func (x *XLAPositionalOps) Forward(shape []int, data ...[]float64) []float64 {
	out, _ := x.RoPEForward(10000.0, shape, data...)
	return out
}

// ---------------------------------------------------------------------------
// ALiBi
// ---------------------------------------------------------------------------

// ALiBiForward computes the ALiBi bias tensor.
// shape=[num_heads, seq_len_q, seq_len_k].
func (x *XLAPositionalOps) ALiBiForward(shape []int, causal bool) ([]float64, error) {
	numHeads := shape[0]
	seqLenQ := shape[1]
	seqLenK := shape[2]

	if err := x.ensureCompiledALiBi(numHeads, seqLenQ, seqLenK); err != nil {
		return nil, err
	}

	slopes := xlaSlopes(numHeads)
	dst := make([]float64, numHeads*seqLenQ*seqLenK)

	causalInt := 0
	if causal {
		causalInt = 1
	}

	rc := C.xla_alibi(
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&slopes[0])),
		C.int(numHeads),
		C.int(seqLenQ),
		C.int(seqLenK),
		C.int(causalInt),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_alibi failed")
	}
	return dst, nil
}
