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
//	shape   = [batch, seq_len]
//	data[0] = flat float64 token IDs, length batch*seq_len
//	data[1] = flat float64 weight table, length VocabSize*DModel
//
// Returns []float64 of length batch*seq_len*DModel.
func (c *CUDAEmbedding) Forward(shape []int, data ...[]float64) []float64 {
	out, err := c.TokenEmbedding(data[0], data[1])
	if err != nil {
		panic(err)
	}

	return out
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
