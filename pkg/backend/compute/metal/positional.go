//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "positional.h"
import "C"

import (
	"fmt"
	"math"
	"strings"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/rotary"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

// MetalPositional dispatches RoPE and ALiBi kernels to the Metal GPU.
type MetalPositional struct {
	metallib string
}

const (
	metalRoPEModeAdjacent = 0
	metalRoPEModeHalf     = 1
)

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
func buildTablesFP32(
	seqLen int,
	headDim int,
	config rotary.Config,
	positionStart int,
) ([]float32, []float32, error) {
	numPairs := headDim / 2
	n := seqLen * numPairs
	cosT := make([]float32, n)
	sinT := make([]float32, n)
	frequencies, err := config.InverseFrequencies(headDim)

	if err != nil {
		return nil, nil, err
	}

	for t := 0; t < seqLen; t++ {
		for i := 0; i < numPairs; i++ {
			theta := frequencies[i]
			angle := float64(positionStart+t) * theta
			cosT[t*numPairs+i] = float32(math.Cos(angle))
			sinT[t*numPairs+i] = float32(math.Sin(angle))
		}
	}
	return cosT, sinT, nil
}

// RoPEForward applies rotary position embeddings on the Metal GPU.
// shape=[batch, num_heads, seq_len, head_dim]; data[0]=input tensor.
func (m *MetalPositional) RoPEForward(base float64, shape []int, data ...[]float64) ([]float64, error) {
	return m.RoPEForwardAt(base, 0, shape, data...)
}

/*
RoPEForwardAt applies rotary position embeddings from an absolute position offset.
*/
func (m *MetalPositional) RoPEForwardAt(
	base float64,
	positionStart int,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return m.RoPEForwardAtMode(base, positionStart, "", shape, data...)
}

/*
RoPEForwardAtMode applies rotary position embeddings with an explicit pair layout.
*/
func (m *MetalPositional) RoPEForwardAtMode(
	base float64,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	return m.RoPEForwardAtModeConfig(
		rotary.Config{Base: base},
		positionStart,
		mode,
		shape,
		data...,
	)
}

/*
RoPEForwardAtModeConfig applies RoPE using a manifest-provided frequency schedule.
*/
func (m *MetalPositional) RoPEForwardAtModeConfig(
	config rotary.Config,
	positionStart int,
	mode string,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("metal_rope: input[0] is required")
	}

	if len(shape) != 4 {
		return nil, fmt.Errorf("metal_rope: expected rank 4 shape, got %d", len(shape))
	}

	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	headDim := shape[3]

	if batch <= 0 || numHeads <= 0 || seqLen <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("metal_rope: all shape dimensions must be positive")
	}

	if headDim%2 != 0 {
		return nil, fmt.Errorf("metal_rope: expected even head_dim, got %d", headDim)
	}

	x := data[0]

	if len(x) != batch*numHeads*seqLen*headDim {
		return nil, fmt.Errorf("metal_rope: input length mismatch")
	}

	src32 := toFloat32(x)
	dst32 := make([]float32, len(x))

	cosT, sinT, err := buildTablesFP32(seqLen, headDim, config, positionStart)

	if err != nil {
		return nil, err
	}

	modeCode, err := metalRoPEMode(mode)

	if err != nil {
		return nil, err
	}

	totalHeads := batch * numHeads
	rc := C.metal_rope(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		(*C.float)(unsafe.Pointer(&cosT[0])),
		(*C.float)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(modeCode),
		C.int(totalHeads),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_rope failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

/*
RoPETensor applies rotary position embeddings without leaving Metal storage.
*/
func (m *MetalPositional) RoPETensor(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	base float64,
	positionStart int,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	return m.RoPETensorMode(
		input,
		outputShape,
		base,
		positionStart,
		"",
		batch, numHeads, seqLen, headDim,
	)
}

/*
RoPETensorMode applies rotary position embeddings with an explicit pair layout.
*/
func (m *MetalPositional) RoPETensorMode(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	base float64,
	positionStart int,
	mode string,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	return m.RoPETensorModeConfig(
		input,
		outputShape,
		rotary.Config{Base: base},
		positionStart,
		mode,
		batch, numHeads, seqLen, headDim,
	)
}

/*
RoPETensorModeConfig applies RoPE in resident Metal storage using a frequency schedule.
*/
func (m *MetalPositional) RoPETensorModeConfig(
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	config rotary.Config,
	positionStart int,
	mode string,
	batch, numHeads, seqLen, headDim int,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if batch <= 0 || numHeads <= 0 || seqLen <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("metal_rope_tensor: all shape dimensions must be positive")
	}

	if headDim%2 != 0 {
		return nil, fmt.Errorf("metal_rope_tensor: expected even head_dim, got %d", headDim)
	}

	expectedLength := batch * numHeads * seqLen * headDim

	if metalInput.Len() != expectedLength || outputShape.Len() != expectedLength {
		return nil, fmt.Errorf("metal_rope_tensor: input/output length mismatch")
	}

	cosT, sinT, err := buildTablesFP32(seqLen, headDim, config, positionStart)

	if err != nil {
		return nil, err
	}

	modeCode, err := metalRoPEMode(mode)

	if err != nil {
		return nil, err
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_rope_tensor(
		metalInput.buffer,
		output.buffer,
		(*C.float)(unsafe.Pointer(&cosT[0])),
		(*C.float)(unsafe.Pointer(&sinT[0])),
		C.int(seqLen),
		C.int(headDim),
		C.int(modeCode),
		C.int(batch*numHeads),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_rope_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func metalRoPEMode(mode string) (int, error) {
	switch strings.ToLower(mode) {
	case "", "adjacent":
		return metalRoPEModeAdjacent, nil
	case "half":
		return metalRoPEModeHalf, nil
	default:
		return 0, fmt.Errorf("metal_rope: unsupported mode %q", mode)
	}
}

// Forward dispatches RoPE with the universal signature.
// shape=[batch, num_heads, seq_len, head_dim]; uses default base=10000.
func (m *MetalPositional) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return m.RoPEForward(10000.0, shape, data...)
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
