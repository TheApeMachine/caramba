//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "attention.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// MetalAttention holds the path to the compiled attention.metallib and manages
// Metal pipeline state for all attention variants.
type MetalAttention struct {
	metallib string
	// Window is used only for SlidingWindow attention.
	Window int
}

// NewAttention creates and initializes a MetalAttention.
// metallib must be the absolute path to attention.metallib compiled from
// attention.metal via:
//
//	xcrun -sdk macosx metal -c attention.metal -o attention.air
//	xcrun -sdk macosx metallib attention.air -o attention.metallib
func NewAttention(metallib string) (*MetalAttention, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_init_attention(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_init_attention failed (rc=%d): check that %q exists and Metal is available", rc, metallib)
	}
	return &MetalAttention{metallib: metallib}, nil
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
func (m *MetalAttention) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 3 {
		return nil, fmt.Errorf(
			"metal attention Forward: expected Q, K, V in data[0..2], got %d slices", len(data),
		)
	}

	switch len(shape) {
	case 5:
		batch, numHeads, numKVHeads, seqLen, headDim :=
			shape[0], shape[1], shape[2], shape[3], shape[4]

		return m.GQA(data[0], data[1], data[2], batch, numHeads, numKVHeads, seqLen, headDim)

	case 4:
		batch, numHeads, seqLen, headDim := shape[0], shape[1], shape[2], shape[3]

		kvSize := batch * 1 * seqLen * headDim

		if len(data[1]) == kvSize {
			return m.MQA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)
		}

		if m.Window > 0 {
			return m.SlidingWindow(data[0], data[1], data[2], batch, numHeads, seqLen, headDim, m.Window)
		}

		return m.SDPA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)

	default:
		return nil, fmt.Errorf("metal attention Forward: unsupported shape rank %d (want 4 or 5)", len(shape))
	}
}

// SDPA computes standard scaled dot-product attention.
func (m *MetalAttention) SDPA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	n := batch * numHeads * seqLen * headDim
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("metal_sdpa: Q/K/V length mismatch (expected %d)", n)
	}

	qf, kf, vf := toFloat32(q), toFloat32(k), toFloat32(v)
	outf := make([]float32, n)

	rc := C.metal_sdpa(
		(*C.float)(unsafe.Pointer(&qf[0])),
		(*C.float)(unsafe.Pointer(&kf[0])),
		(*C.float)(unsafe.Pointer(&vf[0])),
		(*C.float)(unsafe.Pointer(&outf[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_sdpa failed (rc=%d)", rc)
	}
	return toFloat64(outf), nil
}

// MQA computes multi-query attention (K/V shared across Q heads).
func (m *MetalAttention) MQA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	qn := batch * numHeads * seqLen * headDim
	kvn := batch * 1 * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("metal_mqa: input length mismatch")
	}

	qf, kf, vf := toFloat32(q), toFloat32(k), toFloat32(v)
	outf := make([]float32, qn)

	rc := C.metal_mqa(
		(*C.float)(unsafe.Pointer(&qf[0])),
		(*C.float)(unsafe.Pointer(&kf[0])),
		(*C.float)(unsafe.Pointer(&vf[0])),
		(*C.float)(unsafe.Pointer(&outf[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_mqa failed (rc=%d)", rc)
	}
	return toFloat64(outf), nil
}

// GQA computes grouped query attention.
func (m *MetalAttention) GQA(q, k, v []float64, batch, numHeads, numKVHeads, seqLen, headDim int) ([]float64, error) {
	qn := batch * numHeads * seqLen * headDim
	kvn := batch * numKVHeads * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("metal_gqa: input length mismatch")
	}

	qf, kf, vf := toFloat32(q), toFloat32(k), toFloat32(v)
	outf := make([]float32, qn)

	rc := C.metal_gqa(
		(*C.float)(unsafe.Pointer(&qf[0])),
		(*C.float)(unsafe.Pointer(&kf[0])),
		(*C.float)(unsafe.Pointer(&vf[0])),
		(*C.float)(unsafe.Pointer(&outf[0])),
		C.int(batch), C.int(numHeads), C.int(numKVHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_gqa failed (rc=%d)", rc)
	}
	return toFloat64(outf), nil
}

// SlidingWindow computes sliding-window masked attention.
func (m *MetalAttention) SlidingWindow(q, k, v []float64, batch, numHeads, seqLen, headDim, window int) ([]float64, error) {
	n := batch * numHeads * seqLen * headDim
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("metal_sliding_window: Q/K/V length mismatch (expected %d)", n)
	}

	qf, kf, vf := toFloat32(q), toFloat32(k), toFloat32(v)
	outf := make([]float32, n)

	rc := C.metal_sliding_window(
		(*C.float)(unsafe.Pointer(&qf[0])),
		(*C.float)(unsafe.Pointer(&kf[0])),
		(*C.float)(unsafe.Pointer(&vf[0])),
		(*C.float)(unsafe.Pointer(&outf[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim), C.int(window),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_sliding_window failed (rc=%d)", rc)
	}
	return toFloat64(outf), nil
}
