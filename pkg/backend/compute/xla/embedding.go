//go:build cgo && xla

package xla

// XLA token-embedding backend via the PJRT C API.
//
// Build requirements:
//   - XLA headers on the include path (set XLA_INCLUDE via CGO_CPPFLAGS)
//   - PJRT plugin shared library for your platform on LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
//   - Compile embedding_xla.cc alongside this package (CGo picks it up automatically)
//
// Example build:
//   CGO_CPPFLAGS="-I/path/to/xla" \
//   go build -tags "cgo xla" ./pk./pkg/backend/compute/xla

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
	if err := NewPJRTConfig(platform).ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_embedding_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_embedding_init failed for platform %q", platform)
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
func (x *XLAEmbedding) Forward(shape []int, data ...[]float64) []float64 {
	out, err := x.TokenEmbedding(data[0], data[1])
	if err != nil {
		panic(err)
	}

	return out
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
