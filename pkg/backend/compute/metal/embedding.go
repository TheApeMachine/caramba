//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "embedding.h"
import "C"

import (
	"fmt"
	"unsafe"
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
//
// Returns []float64 of length batch*seq_len*DModel.
func (e *EmbeddingOps) Forward(shape []int, data ...[]float64) []float64 {
	out, err := e.TokenEmbedding(data[0], data[1])
	if err != nil {
		panic(err)
	}

	return out
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
