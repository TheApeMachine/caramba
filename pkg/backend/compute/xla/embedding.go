//go:build cgo && xla

package xla

// XLA token-embedding backend via the PJRT C API.
//
// Configure PJRT paths under compute.xla in cmd/asset/config.yml before runtime validation.

// #include <stdlib.h>
// #include "embedding.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAEmbedding dispatches the token embedding lookup to XLA via PJRT.
type XLAEmbedding struct {
	platform  string
	vocabSize int
	dModel    int
}

// NewXLAEmbedding initialises the PJRT client for the given platform
// ("cpu" or "gpu") and stores the embedding dimensions.
func NewXLAEmbedding(platform string, vocabSize, dModel int) (*XLAEmbedding, error) {
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_embedding_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_embedding_init failed for platform %q: rc=%d", platform, rc)
	}
	return &XLAEmbedding{platform: platform, vocabSize: vocabSize, dModel: dModel}, nil
}

// Shutdown releases all PJRT embedding resources.
func (x *XLAEmbedding) Shutdown() {
	C.xla_embedding_shutdown()
}

// Forward performs token embedding lookup.
//
//	shape   = [batch, seq_len]
//	data[0] = flat float64 token IDs, length batch*seq_len
//	data[1] = flat float64 weight table, length VocabSize*DModel
//
// Returns []float64 of length batch*seq_len*DModel.
func (x *XLAEmbedding) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf(
			"xla embedding Forward: expected tokens and weights (2 buffers), got %d",
			len(data),
		)
	}

	if len(shape) != 2 {
		return nil, fmt.Errorf(
			"xla embedding Forward: expected shape [batch, seq_len], got %v",
			shape,
		)
	}

	batch, seqLen := shape[0], shape[1]
	wantTokens := batch * seqLen

	if len(data[0]) != wantTokens {
		return nil, fmt.Errorf(
			"xla embedding Forward: len(tokens)=%d, expected batch*seq_len=%d*%d=%d",
			len(data[0]), batch, seqLen, wantTokens,
		)
	}

	wantWeights := x.vocabSize * x.dModel

	if len(data[1]) != wantWeights {
		return nil, fmt.Errorf(
			"xla embedding Forward: len(weights)=%d, expected vocab*d_model=%d*%d=%d",
			len(data[1]), x.vocabSize, x.dModel, wantWeights,
		)
	}

	return x.TokenEmbedding(data[0], data[1])
}

// TokenEmbedding performs the lookup given token IDs and the weight table.
func (x *XLAEmbedding) TokenEmbedding(tokens []float64, weight []float64) ([]float64, error) {
	n := len(tokens)
	if n == 0 {
		return []float64{}, nil
	}

	// Ensure compiled for the current dimensions.
	if rc := C.xla_compile_embedding(C.int(n), C.int(x.dModel), C.int(x.vocabSize)); rc != 0 {
		return nil, fmt.Errorf("xla_compile_embedding(%d, %d, %d) failed", n, x.dModel, x.vocabSize)
	}

	out := make([]float64, n*x.dModel)
	rc := C.xla_token_embedding(
		(*C.double)(unsafe.Pointer(&tokens[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		C.int(n),
		C.int(x.dModel),
		C.int(x.vocabSize),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_token_embedding failed (rc=%d)", rc)
	}
	return out, nil
}
