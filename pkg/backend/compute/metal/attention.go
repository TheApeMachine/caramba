//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "attention.h"
import "C"

import (
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
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

		return m.GQA(data[0], data[1], data[2], batch, numHeads, numKVHeads, seqLen, headDim, false)

	case 4:
		batch, numHeads, seqLen, headDim := shape[0], shape[1], shape[2], shape[3]

		kvSize := batch * 1 * seqLen * headDim

		if len(data[1]) == kvSize {
			return m.MQA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim)
		}

		if m.Window > 0 {
			return m.SlidingWindow(data[0], data[1], data[2], batch, numHeads, seqLen, headDim, m.Window)
		}

		return m.SDPA(data[0], data[1], data[2], batch, numHeads, seqLen, headDim, false)

	default:
		return nil, fmt.Errorf("metal attention Forward: unsupported shape rank %d (want 4 or 5)", len(shape))
	}
}

// SDPA computes standard scaled dot-product attention.
func (m *MetalAttention) SDPA(
	q, k, v []float64,
	batch, numHeads, queryLen, headDim int,
	causal bool,
) ([]float64, error) {
	qn := batch * numHeads * queryLen * headDim
	keyValueWidth := batch * numHeads * headDim

	if len(q) != qn {
		return nil, fmt.Errorf("metal_sdpa: Q length mismatch (expected %d)", qn)
	}

	if len(k) != len(v) || keyValueWidth == 0 || len(k)%keyValueWidth != 0 {
		return nil, fmt.Errorf("metal_sdpa: K/V lengths must match whole cached heads")
	}

	keyValueLen := len(k) / keyValueWidth

	if keyValueLen < queryLen {
		return nil, fmt.Errorf(
			"metal_sdpa: key/value length %d is shorter than query length %d",
			keyValueLen,
			queryLen,
		)
	}

	qf, kf, vf := toFloat32(q), toFloat32(k), toFloat32(v)
	outf := make([]float32, qn)
	causalInt := 0

	if causal {
		causalInt = 1
	}

	rc := C.metal_sdpa(
		(*C.float)(unsafe.Pointer(&qf[0])),
		(*C.float)(unsafe.Pointer(&kf[0])),
		(*C.float)(unsafe.Pointer(&vf[0])),
		(*C.float)(unsafe.Pointer(&outf[0])),
		C.int(batch),
		C.int(numHeads),
		C.int(queryLen),
		C.int(keyValueLen),
		C.int(headDim),
		C.int(causalInt),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_sdpa failed (rc=%d)", rc)
	}
	return toFloat64(outf), nil
}

/*
SDPATensor computes scaled dot-product attention without leaving Metal storage.
*/
func (m *MetalAttention) SDPATensor(
	q, k, v computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch, numHeads, queryLen, keyValueLen, keyValueStride, headDim int,
	causal bool,
) (computetensor.Float64Tensor, error) {
	metalQ, err := requireMetalTensor(q)

	if err != nil {
		return nil, err
	}

	metalK, err := requireMetalTensor(k)

	if err != nil {
		return nil, err
	}

	metalV, err := requireMetalTensor(v)

	if err != nil {
		return nil, err
	}

	queryLength := batch * numHeads * queryLen * headDim
	keyValueLength := batch * numHeads * keyValueStride * headDim

	if queryLength <= 0 || keyValueLength <= 0 || keyValueStride < keyValueLen {
		return nil, fmt.Errorf("metal_sdpa_tensor: dimensions must be positive")
	}

	if metalQ.Len() != queryLength {
		return nil, fmt.Errorf("metal_sdpa_tensor: Q length mismatch")
	}

	if metalK.Len() != keyValueLength || metalV.Len() != keyValueLength {
		return nil, fmt.Errorf("metal_sdpa_tensor: K/V length mismatch")
	}

	if outputShape.Len() != queryLength {
		return nil, fmt.Errorf("metal_sdpa_tensor: output length mismatch")
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	causalInt := 0

	if causal {
		causalInt = 1
	}

	rc := C.metal_sdpa_tensor(
		metalQ.buffer,
		metalK.buffer,
		metalV.buffer,
		output.buffer,
		C.int(batch),
		C.int(numHeads),
		C.int(queryLen),
		C.int(keyValueLen),
		C.int(keyValueStride),
		C.int(headDim),
		C.int(causalInt),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_sdpa_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
RepackKVTensor copies live cache tokens into a larger capacity buffer.
*/
func (m *MetalAttention) RepackKVTensor(
	previousKey computetensor.Float64Tensor,
	previousValue computetensor.Float64Tensor,
	outputKey computetensor.Float64Tensor,
	outputValue computetensor.Float64Tensor,
	batch, numHeads, currentLen, headDim, previousCapacity, outputCapacity int,
) error {
	metalPreviousKey, err := requireMetalTensor(previousKey)

	if err != nil {
		return err
	}

	metalPreviousValue, err := requireMetalTensor(previousValue)

	if err != nil {
		return err
	}

	metalOutputKey, err := requireMetalTensor(outputKey)

	if err != nil {
		return err
	}

	metalOutputValue, err := requireMetalTensor(outputValue)

	if err != nil {
		return err
	}

	previousLength := batch * numHeads * previousCapacity * headDim
	outputLength := batch * numHeads * outputCapacity * headDim

	if currentLen <= 0 || previousCapacity < currentLen || outputCapacity < currentLen {
		return fmt.Errorf("metal_kv_repack_tensor: invalid cache dimensions")
	}

	if metalPreviousKey.Len() != previousLength || metalPreviousValue.Len() != previousLength ||
		metalOutputKey.Len() != outputLength || metalOutputValue.Len() != outputLength {
		return fmt.Errorf("metal_kv_repack_tensor: cache length mismatch")
	}

	rc := C.metal_kv_repack_tensor(
		metalPreviousKey.buffer,
		metalPreviousValue.buffer,
		metalOutputKey.buffer,
		metalOutputValue.buffer,
		C.int(batch),
		C.int(numHeads),
		C.int(currentLen),
		C.int(headDim),
		C.int(previousCapacity),
		C.int(outputCapacity),
	)

	if rc != 0 {
		return fmt.Errorf("metal_kv_repack_tensor failed (rc=%d)", rc)
	}

	return nil
}

/*
WriteKVTensor writes a K/V chunk into an existing capacity-backed cache.
*/
func (m *MetalAttention) WriteKVTensor(
	cacheKey computetensor.Float64Tensor,
	cacheValue computetensor.Float64Tensor,
	keyChunk computetensor.Float64Tensor,
	valueChunk computetensor.Float64Tensor,
	batch, numHeads, startLen, chunkLen, headDim, capacity int,
) error {
	metalCacheKey, err := requireMetalTensor(cacheKey)

	if err != nil {
		return err
	}

	metalCacheValue, err := requireMetalTensor(cacheValue)

	if err != nil {
		return err
	}

	metalChunkKey, err := requireMetalTensor(keyChunk)

	if err != nil {
		return err
	}

	metalChunkValue, err := requireMetalTensor(valueChunk)

	if err != nil {
		return err
	}

	cacheLength := batch * numHeads * capacity * headDim
	chunkLength := batch * numHeads * chunkLen * headDim

	if startLen < 0 || chunkLen <= 0 || capacity < startLen+chunkLen {
		return fmt.Errorf("metal_kv_write_tensor: invalid cache dimensions")
	}

	if metalCacheKey.Len() != cacheLength || metalCacheValue.Len() != cacheLength ||
		metalChunkKey.Len() != chunkLength || metalChunkValue.Len() != chunkLength {
		return fmt.Errorf("metal_kv_write_tensor: cache length mismatch")
	}

	rc := C.metal_kv_write_tensor(
		metalCacheKey.buffer,
		metalCacheValue.buffer,
		metalChunkKey.buffer,
		metalChunkValue.buffer,
		C.int(batch),
		C.int(numHeads),
		C.int(startLen),
		C.int(chunkLen),
		C.int(headDim),
		C.int(capacity),
	)

	if rc != 0 {
		return fmt.Errorf("metal_kv_write_tensor failed (rc=%d)", rc)
	}

	return nil
}

/*
AppendKVTensor appends a K/V chunk along the token dimension in resident Metal storage.
*/
func (m *MetalAttention) AppendKVTensor(
	previousKey computetensor.Float64Tensor,
	previousValue computetensor.Float64Tensor,
	keyChunk computetensor.Float64Tensor,
	valueChunk computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch, numHeads, previousLen, chunkLen, headDim int,
) (computetensor.Float64Tensor, computetensor.Float64Tensor, error) {
	metalChunkKey, err := requireMetalTensor(keyChunk)

	if err != nil {
		return nil, nil, err
	}

	metalChunkValue, err := requireMetalTensor(valueChunk)

	if err != nil {
		return nil, nil, err
	}

	var metalPreviousKey *Tensor
	var metalPreviousValue *Tensor

	if previousLen > 0 {
		metalPreviousKey, err = requireMetalTensor(previousKey)

		if err != nil {
			return nil, nil, err
		}

		metalPreviousValue, err = requireMetalTensor(previousValue)

		if err != nil {
			return nil, nil, err
		}
	}

	chunkLength := batch * numHeads * chunkLen * headDim
	previousLength := batch * numHeads * previousLen * headDim

	if chunkLength <= 0 || previousLength < 0 {
		return nil, nil, fmt.Errorf("metal_kv_append_tensor: invalid cache dimensions")
	}

	if metalChunkKey.Len() != chunkLength || metalChunkValue.Len() != chunkLength {
		return nil, nil, fmt.Errorf("metal_kv_append_tensor: chunk length mismatch")
	}

	if previousLen > 0 &&
		(metalPreviousKey.Len() != previousLength || metalPreviousValue.Len() != previousLength) {
		return nil, nil, fmt.Errorf("metal_kv_append_tensor: previous length mismatch")
	}

	if outputShape.Len() != previousLength+chunkLength {
		return nil, nil, fmt.Errorf("metal_kv_append_tensor: output length mismatch")
	}

	outputKey, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, nil, err
	}

	outputValue, err := newMetalTensor(outputShape)

	if err != nil {
		_ = outputKey.Close()

		return nil, nil, err
	}

	var previousKeyBuffer unsafe.Pointer
	var previousValueBuffer unsafe.Pointer

	if previousLen > 0 {
		previousKeyBuffer = metalPreviousKey.buffer
		previousValueBuffer = metalPreviousValue.buffer
	}

	rc := C.metal_kv_append_tensor(
		previousKeyBuffer,
		previousValueBuffer,
		metalChunkKey.buffer,
		metalChunkValue.buffer,
		outputKey.buffer,
		outputValue.buffer,
		C.int(batch),
		C.int(numHeads),
		C.int(previousLen),
		C.int(chunkLen),
		C.int(headDim),
	)

	if rc != 0 {
		_ = outputKey.Close()
		_ = outputValue.Close()

		return nil, nil, fmt.Errorf("metal_kv_append_tensor failed (rc=%d)", rc)
	}

	return outputKey, outputValue, nil
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
func (m *MetalAttention) GQA(
	q, k, v []float64,
	batch, numHeads, numKVHeads, seqLen, headDim int,
	causal bool,
) ([]float64, error) {
	qn := batch * numHeads * seqLen * headDim
	kvn := batch * numKVHeads * seqLen * headDim
	if len(q) != qn || len(k) != kvn || len(v) != kvn {
		return nil, fmt.Errorf("metal_gqa: input length mismatch")
	}

	qf, kf, vf := toFloat32(q), toFloat32(k), toFloat32(v)
	outf := make([]float32, qn)
	causalFlag := 0

	if causal {
		causalFlag = 1
	}

	rc := C.metal_gqa(
		(*C.float)(unsafe.Pointer(&qf[0])),
		(*C.float)(unsafe.Pointer(&kf[0])),
		(*C.float)(unsafe.Pointer(&vf[0])),
		(*C.float)(unsafe.Pointer(&outf[0])),
		C.int(batch), C.int(numHeads), C.int(numKVHeads), C.int(seqLen), C.int(headDim),
		C.int(causalFlag),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_gqa failed (rc=%d)", rc)
	}
	return toFloat64(outf), nil
}

/*
GQATensor computes grouped-query attention without leaving Metal storage.
*/
func (m *MetalAttention) GQATensor(
	q, k, v computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch, numHeads, numKVHeads, queryLen, keyValueLen, keyValueStride, headDim int,
	causal bool,
) (computetensor.Float64Tensor, error) {
	metalQ, err := requireMetalTensor(q)

	if err != nil {
		return nil, err
	}

	metalK, err := requireMetalTensor(k)

	if err != nil {
		return nil, err
	}

	metalV, err := requireMetalTensor(v)

	if err != nil {
		return nil, err
	}

	queryLength := batch * numHeads * queryLen * headDim
	keyValueLength := batch * numKVHeads * keyValueStride * headDim

	if queryLength <= 0 || keyValueLength <= 0 || keyValueStride < keyValueLen ||
		keyValueLen < queryLen || numHeads%numKVHeads != 0 {
		return nil, fmt.Errorf("metal_gqa_tensor: dimensions must be positive")
	}

	if metalQ.Len() != queryLength {
		return nil, fmt.Errorf("metal_gqa_tensor: Q length mismatch")
	}

	if metalK.Len() != keyValueLength || metalV.Len() != keyValueLength {
		return nil, fmt.Errorf("metal_gqa_tensor: K/V length mismatch")
	}

	if outputShape.Len() != queryLength {
		return nil, fmt.Errorf("metal_gqa_tensor: output length mismatch")
	}

	output, err := newMetalTensor(outputShape)

	if err != nil {
		return nil, err
	}

	causalFlag := 0

	if causal {
		causalFlag = 1
	}

	rc := C.metal_gqa_tensor(
		metalQ.buffer,
		metalK.buffer,
		metalV.buffer,
		output.buffer,
		C.int(batch),
		C.int(numHeads),
		C.int(numKVHeads),
		C.int(queryLen),
		C.int(keyValueLen),
		C.int(keyValueStride),
		C.int(headDim),
		C.int(causalFlag),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_gqa_tensor failed (rc=%d)", rc)
	}

	return output, nil
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
