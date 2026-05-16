//go:build cgo && xla

package xla

// XLA positional backend via the PJRT C API.
//
// Configure PJRT paths under compute.xla in cmd/asset/config.yml before runtime validation.

// #include <stdlib.h>
// #include "positional.h"
import "C"

import (
	"fmt"
	"math"
	"strings"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
)

// XLAPositionalOps dispatches RoPE and ALiBi to the XLA runtime via PJRT.
type XLAPositionalOps struct {
	platform string
}

const (
	xlaRoPEModeAdjacent = 0
	xlaRoPEModeHalf     = 1
)

// NewPositional initialises the PJRT client for the given platform.
func NewPositional(platform string) (*XLAPositionalOps, error) {
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_positional_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_positional_init failed for platform %q", config.Platform)
	}

	return &XLAPositionalOps{platform: config.Platform}, nil
}

// Shutdown releases all PJRT resources.
func (x *XLAPositionalOps) Shutdown() {
	C.xla_positional_shutdown()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func xlaCosSinTables(
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
	return x.RoPEForwardAt(base, 0, shape, data...)
}

func (x *XLAPositionalOps) RoPEForwardAt(
	base float64,
	positionStart int,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return x.RoPEForwardAtMode(base, positionStart, "", shape, data...)
}

func (x *XLAPositionalOps) RoPEForwardAtMode(
	base float64,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return x.RoPEForwardAtModeConfig(
		rotary.Config{Base: base},
		positionStart,
		mode,
		shape,
		data...,
	)
}

func (x *XLAPositionalOps) RoPEForwardAtModeConfig(
	config rotary.Config,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("xla_rope: input[0] is required")
	}

	if len(shape) != 4 {
		return nil, fmt.Errorf("xla_rope: expected rank 4 shape, got %d", len(shape))
	}

	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	if batch <= 0 || numHeads <= 0 || seqLen <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("xla_rope: all shape dimensions must be positive")
	}

	if headDim%2 != 0 {
		return nil, fmt.Errorf("xla_rope: expected even head_dim, got %d", headDim)
	}

	totalHeads := batch * numHeads

	input := data[0]

	if len(input) != totalHeads*seqLen*headDim {
		return nil, fmt.Errorf("xla_rope: input length mismatch")
	}

	if err := x.ensureCompiledRoPE(totalHeads, seqLen, headDim); err != nil {
		return nil, err
	}

	dst := make([]float64, len(input))
	cosT, sinT, err := xlaCosSinTables(seqLen, headDim, config, positionStart)

	if err != nil {
		return nil, err
	}

	modeCode, err := xlaRoPEMode(mode)

	if err != nil {
		return nil, err
	}

	rc := C.xla_rope(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&cosT[0])),
		(*C.double)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(modeCode),
		C.int(totalHeads),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_rope failed")
	}
	return dst, nil
}

func xlaRoPEMode(mode string) (int, error) {
	switch strings.ToLower(mode) {
	case "", "adjacent":
		return xlaRoPEModeAdjacent, nil
	case "half":
		return xlaRoPEModeHalf, nil
	default:
		return 0, fmt.Errorf("xla_rope: unsupported mode %q", mode)
	}
}

// Forward dispatches RoPEForward with base 0 so RoPEForward applies the
// standard RoPE default base of 10000.0.
func (x *XLAPositionalOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return x.RoPEForward(0, shape, data...)
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
