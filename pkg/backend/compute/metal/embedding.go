//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "embedding.h"
import "C"

import (
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

// EmbeddingOps wraps the Metal token-embedding kernel.
type EmbeddingOps struct {
	metallib  string
	vocabSize int
	dModel    int
}

// NewEmbeddingOps creates an EmbeddingOps and initialises the Metal pipeline.
// metallib must be the absolute path to embedding.metallib compiled from
// embedding.metal via:
//
//	xcrun -sdk macosx metal -c embedding.metal -o embedding.air
//	xcrun -sdk macosx metallib embedding.air -o embedding.metallib
func NewEmbeddingOps(metallib string, vocabSize, dModel int) (*EmbeddingOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_embedding_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_embedding_init failed (rc=%d): check that %q exists", rc, metallib)
	}
	return &EmbeddingOps{metallib: metallib, vocabSize: vocabSize, dModel: dModel}, nil
}

// Forward performs token embedding lookup.
//
//	shape   = [batch, seq_len]
//	data[0] = flat float64 token IDs, length batch*seq_len
//	data[1] = embedding weight table, length vocabSize*dModel
//
// Returns []float64 of length batch*seq_len*DModel.
func (e *EmbeddingOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf(
			"metal embedding Forward: expected data[0] tokens and data[1] weights, got %d slices",
			len(data),
		)
	}

	if len(shape) < 2 {
		return nil, fmt.Errorf("metal embedding Forward: shape must be [batch, seq_len], got len %d", len(shape))
	}

	batch, seqLen := shape[0], shape[1]
	expectedTok := batch * seqLen

	if expectedTok < 0 {
		return nil, fmt.Errorf("metal embedding Forward: invalid batch*seq_len")
	}

	if len(data[0]) != expectedTok {
		return nil, fmt.Errorf(
			"metal embedding Forward: token slice length %d != batch*seq_len %d",
			len(data[0]), expectedTok,
		)
	}

	expectedWeight := e.vocabSize * e.dModel

	if len(data[1]) != expectedWeight {
		return nil, fmt.Errorf(
			"metal embedding Forward: weight slice length %d != vocabSize*dModel %d",
			len(data[1]), expectedWeight,
		)
	}

	return e.TokenEmbedding(data[0], data[1])
}

// TokenEmbedding performs the lookup given token IDs and the weight table.
//   - tokens: float64 token IDs, length batch*seq_len
//   - weight: flat weight table float64, length vocabSize*dModel
//
// Returns (output, error).
func (e *EmbeddingOps) TokenEmbedding(tokens []float64, weight []float64) ([]float64, error) {
	batchSeq := len(tokens)
	if batchSeq == 0 {
		return []float64{}, nil
	}

	// Convert float64 → float32 for Metal (float32 kernel).
	tokF32 := toFloat32(tokens)
	wF32 := toFloat32(weight)
	outF32 := make([]float32, batchSeq*e.dModel)

	rc := C.metal_token_embedding(
		(*C.float)(unsafe.Pointer(&tokF32[0])),
		(*C.float)(unsafe.Pointer(&outF32[0])),
		(*C.float)(unsafe.Pointer(&wF32[0])),
		C.int(batchSeq),
		C.int(e.dModel),
		C.int(e.vocabSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_token_embedding failed (rc=%d)", rc)
	}
	return toFloat64(outF32), nil
}

/*
ForwardTensor performs token embedding lookup against resident Metal buffers.
*/
func (e *EmbeddingOps) ForwardTensor(
	tokens computetensor.Float64Tensor,
	weight computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (computetensor.Float64Tensor, error) {
	metalTokens, err := requireMetalTensor(tokens)

	if err != nil {
		return nil, err
	}

	metalWeight, err := requireMetalTensor(weight)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() != metalTokens.Len()*e.dModel {
		return nil, fmt.Errorf(
			"metal embedding tensor: output shape length %d does not match token_count*d_model=%d",
			outputShape.Len(),
			metalTokens.Len()*e.dModel,
		)
	}

	if metalWeight.Len() != e.vocabSize*e.dModel {
		return nil, fmt.Errorf(
			"metal embedding tensor: weight length %d != vocab_size*d_model=%d",
			metalWeight.Len(),
			e.vocabSize*e.dModel,
		)
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	rc := C.metal_token_embedding_tensor(
		metalTokens.buffer,
		output.buffer,
		metalWeight.buffer,
		C.int(metalTokens.Len()),
		C.int(e.dModel),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_token_embedding_tensor failed (rc=%d)", rc)
	}

	return output, nil
}
