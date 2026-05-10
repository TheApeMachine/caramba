//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "attention.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAAttention dispatches attention kernels to the GPU via CUDA.
// Device memory is managed internally by the C wrappers.
// For SlidingWindow attention set the Window field before calling Forward.
type CUDAAttention struct {
	// Window is the one-sided context radius for sliding-window attention.
	Window int
}

// NewAttention creates a CUDAAttention. No explicit initialization is required.
func NewAttention() *CUDAAttention {
	return &CUDAAttention{}
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
func (c *CUDAAttention) Forward(shape []int, data ...[]float64) []float64 {
	switch len(shape) {
	case 5:
		batch, numHeads, numKVHeads, seqLen, headDim :=
			shape[0], shape[1], shape[2], shape[3], shape[4]
		out, err := c.GQA(data[0], data[1], data[2], batch, numHeads, numKVHeads, seqLen, headDim)
		if err != nil {
			return nil
		}
		return out

	default:
		batch, numHeads, seqLen, headDim := shape[0], shape[1], shape[2], shape[3]
		kvSize := batch * 1 * seqLen * headDim
		if len(data[1]) == kvSize {
			out, err := c.MQA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)
			if err != nil {
				return nil
			}
			return out
		}
		if c.Window > 0 {
			out, err := c.SlidingWindow(data[0], data[1], data[2], batch, numHeads, seqLen, headDim, c.Window)
			if err != nil {
				return nil
			}
			return out
		}
		out, err := c.SDPA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)
		if err != nil {
			return nil
		}
		return out
	}
}

// SDPA computes standard scaled dot-product attention.
func (c *CUDAAttention) SDPA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	n := batch * numHeads * seqLen * headDim
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("cuda_sdpa: Q/K/V length mismatch (expected %d)", n)
	}
	out := make([]float64, n)
	rc := C.cuda_sdpa(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_sdpa failed (rc=%d)", rc)
	}
	return out, nil
}

// MQA computes multi-query attention (K/V shared across Q heads per batch).
func (c *CUDAAttention) MQA(q, k, v []float64, batch, numHeads, seqLen, headDim int) ([]float64, error) {
	qn  := batch * numHeads * seqLen * headDim
	kvn := batch * 1 * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("cuda_mqa: input length mismatch")
	}
	out := make([]float64, qn)
	rc := C.cuda_mqa(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_mqa failed (rc=%d)", rc)
	}
	return out, nil
}

// GQA computes grouped query attention.
func (c *CUDAAttention) GQA(q, k, v []float64, batch, numHeads, numKVHeads, seqLen, headDim int) ([]float64, error) {
	qn  := batch * numHeads * seqLen * headDim
	kvn := batch * numKVHeads * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("cuda_gqa: input length mismatch")
	}
	out := make([]float64, qn)
	rc := C.cuda_gqa(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(numKVHeads), C.int(seqLen), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_gqa failed (rc=%d)", rc)
	}
	return out, nil
}

// SlidingWindow computes sliding-window masked attention.
func (c *CUDAAttention) SlidingWindow(q, k, v []float64, batch, numHeads, seqLen, headDim, window int) ([]float64, error) {
	n := batch * numHeads * seqLen * headDim
	if len(q) != n || len(k) != n || len(v) != n {
		return nil, fmt.Errorf("cuda_sliding_window: Q/K/V length mismatch (expected %d)", n)
	}
	out := make([]float64, n)
	rc := C.cuda_sliding_window(
		(*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&k[0])),
		(*C.double)(unsafe.Pointer(&v[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(numHeads), C.int(seqLen), C.int(headDim), C.int(window),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_sliding_window failed (rc=%d)", rc)
	}
	return out, nil
}
