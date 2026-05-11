//go:build cgo && xla

package xla

// XLA attention backend via the PJRT C API.
//
// Build requirements:
//   - XLA headers on the include path (set XLA_INCLUDE via CGO_CPPFLAGS)
//   - PJRT plugin shared library for your platform on LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
//   - Compile attention_xla.cc alongside this package (CGo picks it up automatically)
//
// Example build:
//   CGO_CPPFLAGS="-I/path/to/xla" \
//   go build -tags "cgo xla" ./pk./pkg/backend/compute/xla

// #include <stdlib.h>
// #include "activation.h"
// #include "attention.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAAttention dispatches attention operations to the XLA runtime via PJRT.
// Compiled executables are cached by shape parameters.
// For SlidingWindow attention set the Window field before calling Forward.
type XLAAttention struct {
	platform string
	// Window is the one-sided context radius for sliding-window attention.
	Window int
}

// NewAttention initialises a new XLAAttention for the given platform ("cpu" or "gpu").
// It reuses the shared PJRT client when one already exists.
func NewAttention(platform string) (*XLAAttention, error) {
	if err := NewPJRTConfig(platform).ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer func() { C.free(unsafe.Pointer(cp)) }()

	if rc := C.xla_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_init failed for platform %q", platform)
	}

	return &XLAAttention{platform: platform}, nil
}

// Forward dispatches to the appropriate attention variant based on len(shape).
//
//   - len(shape)==4: [batch, num_heads, seq_len, head_dim]
//     data[0]=Q, data[1]=K, data[2]=V
//     If Window>0 uses SlidingWindow, otherwise SDPA.
//     If len(K) == batch*1*seq_len*head_dim (< Q), uses MQA.
//
//   - len(shape)==5: [batch, num_heads, num_kv_heads, seq_len, head_dim]
//     data[0]=Q, data[1]=K, data[2]=V — GQA.
func (x *XLAAttention) Forward(shape []int, data ...[]float64) []float64 {
	switch len(shape) {
	case 5:
		batch, numHeads, numKVHeads, seqLen, headDim :=
			shape[0], shape[1], shape[2], shape[3], shape[4]
		out, err := x.GQA(data[0], data[1], data[2], batch, numHeads, numKVHeads, seqLen, headDim)
		if err != nil {
			panic(err)
		}
		return out

	default:
		batch, numHeads, seqLen, headDim := shape[0], shape[1], shape[2], shape[3]
		kvSize := batch * 1 * seqLen * headDim
		if len(data[1]) == kvSize {
			out, err := x.MQA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)
			if err != nil {
				panic(err)
			}
			return out
		}
		if x.Window > 0 {
			out, err := x.SlidingWindow(data[0], data[1], data[2], batch, numHeads, seqLen, headDim, x.Window)
			if err != nil {
				panic(err)
			}
			return out
		}
		out, err := x.SDPA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)
		if err != nil {
			panic(err)
		}
		return out
	}
}

// SDPA computes standard scaled dot-product attention.
func (x *XLAAttention) SDPA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	n := batch * numHeads * seqLen * headDim
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("xla_sdpa: Q/K/V length mismatch (expected %d)", n)
	}
	out := make([]float64, n)
	rc := C.xla_sdpa(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_sdpa failed")
	}
	return out, nil
}

// MQA computes multi-query attention (K/V shared across Q heads per batch).
func (x *XLAAttention) MQA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	qn := batch * numHeads * seqLen * headDim
	kvn := batch * 1 * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("xla_mqa: input length mismatch")
	}
	out := make([]float64, qn)
	rc := C.xla_mqa(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_mqa failed")
	}
	return out, nil
}

// GQA computes grouped query attention.
func (x *XLAAttention) GQA(q, k, v []float64, batch, numHeads, numKVHeads, seqLen, headDim int) ([]float64, error) {
	qn := batch * numHeads * seqLen * headDim
	kvn := batch * numKVHeads * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("xla_gqa: input length mismatch")
	}
	out := make([]float64, qn)
	rc := C.xla_gqa(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(numKVHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_gqa failed")
	}
	return out, nil
}

// SlidingWindow computes sliding-window masked attention.
func (x *XLAAttention) SlidingWindow(q, k, v []float64, batch, numHeads, seqLen, headDim, window int) ([]float64, error) {
	n := batch * numHeads * seqLen * headDim
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("xla_sliding_window: Q/K/V length mismatch (expected %d)", n)
	}
	out := make([]float64, n)
	rc := C.xla_sliding_window(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim), C.int(window),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_sliding_window failed")
	}
	return out, nil
}
