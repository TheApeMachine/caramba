//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "embedding.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAEmbedding dispatches the token embedding lookup to the GPU via CUDA.
type CUDAEmbedding struct {
	VocabSize int
	DModel    int
}

// NewCUDAEmbedding creates a CUDAEmbedding. No explicit initialization is
// required — the CUDA runtime initialises lazily.
func NewCUDAEmbedding(vocabSize, dModel int) *CUDAEmbedding {
	return &CUDAEmbedding{VocabSize: vocabSize, DModel: dModel}
}

// Forward performs token embedding lookup.
//
//	metadata shape = [batch, seq_len] when provided; product(shape) must equal len(data[0])
//	data[0] = flat float64 token IDs, length batch*seq_len
//	data[1] = flat float64 weight table, length VocabSize*DModel
//
// Returns []float64 of length batch*seq_len*DModel.
func (c *CUDAEmbedding) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf(
			"cuda embedding: Forward requires token IDs (data[0]) and weight table (data[1])",
		)
	}

	tokens := data[0]
	weights := data[1]

	if len(shape) > 0 {
		product := 1

		for _, dimension := range shape {
			if dimension < 0 {
				return nil, fmt.Errorf("cuda embedding: negative shape dimension %d", dimension)
			}

			product *= dimension
		}

		if product != len(tokens) {
			return nil, fmt.Errorf(
				"cuda embedding: shape product %d does not match len(tokens)=%d",
				product, len(tokens),
			)
		}
	}

	return c.TokenEmbedding(tokens, weights)
}

// TokenEmbedding performs the lookup given token IDs and the weight table.
func (c *CUDAEmbedding) TokenEmbedding(tokens []float64, weight []float64) ([]float64, error) {
	n := len(tokens)
	if n == 0 {
		return []float64{}, nil
	}
	out := make([]float64, n*c.DModel)

	rc := C.cuda_token_embedding(
		(*C.double)(unsafe.Pointer(&tokens[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		C.int(n),
		C.int(c.DModel),
		C.int(c.VocabSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_token_embedding failed (rc=%d)", rc)
	}
	return out, nil
}
