//go:build darwin && cgo

package metal

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend owns Metal MTLBuffer tensors.
*/
type TensorBackend struct {
	closed          atomic.Uint32
	cacheMu         sync.Mutex
	runtime         *MetalRuntime
	activationOps   *MetalActivation
	mathOps         *MathOps
	shapeOps        *MetalShapeOps
	attentionOps    *MetalAttention
	positionalOps   *MetalPositional
	convolutionOps  *ConvolutionOps
	poolingOps      *PoolingOps
	maskingOps      *MetalMasking
	projectionOps   *ProjectionOps
	vsaOps          *MetalVSAOps
	hawkesOps       *MetalHawkes
	activeOps       *ActiveInferenceOps
	predictiveOps   *MetalPredictiveCodingOps
	markovOps       *MetalMarkovBlanket
	causalOps       *MetalCausalOps
	embeddingOps    map[string]*EmbeddingOps
	kvEntries       map[string]*residentKVEntry
	resident        map[string]residentTensorEntry
	optimizerStates map[string]*residentOptimizerState
}

type residentKVEntry struct {
	epoch    uint64
	capacity int
	shape    []int
	key      computetensor.Tensor
	value    computetensor.Tensor
}

type residentTensorEntry struct {
	shape       []int
	fingerprint uint64
	value       computetensor.Tensor
}

/*
NewTensorBackend creates a Metal resident tensor backend.
*/
func NewTensorBackend() (*TensorBackend, error) {
	runtime, err := NewMetalRuntime(DefaultMetalRuntimeConfig())

	if err != nil {
		return nil, err
	}

	return &TensorBackend{
		runtime:         runtime,
		embeddingOps:    make(map[string]*EmbeddingOps),
		kvEntries:       make(map[string]*residentKVEntry),
		resident:        make(map[string]residentTensorEntry),
		optimizerStates: make(map[string]*residentOptimizerState),
	}, nil
}

/*
Location identifies Metal storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.Metal
}

func (tensorBackend *TensorBackend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{dtype.Float32, dtype.Float64}
}

func (tensorBackend *TensorBackend) SupportedLayouts() []computetensor.Layout {
	return []computetensor.Layout{computetensor.LayoutDense}
}

func (tensorBackend *TensorBackend) Capabilities() computetensor.Capabilities {
	return computetensor.Capabilities{
		MaxBytes:        computetensor.MaxBytesUnlimited,
		SupportsAsync:   false,
		SupportsSparse:  false,
		NativeAlignment: 128,
		NUMANodes:       1,
	}
}

func (tensorBackend *TensorBackend) Upload(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (computetensor.Tensor, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, computetensor.ErrBackendClosed
	}

	if !shape.Valid() {
		return nil, computetensor.ErrShapeInvalid
	}

	expected, err := shape.Bytes(sourceDType)
	if err != nil {
		return nil, err
	}

	if expected != len(bytes) {
		return nil, computetensor.ErrShapeMismatch
	}

	return tensorBackend.runtime.Upload(shape, sourceDType, bytes)
}

func (tensorBackend *TensorBackend) UploadAsync(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (computetensor.Tensor, error) {
	return tensorBackend.Upload(shape, sourceDType, bytes)
}

func (tensorBackend *TensorBackend) UploadSparse(
	shape computetensor.Shape,
	valueDType dtype.DType,
	layout computetensor.Layout,
	values []byte,
	indices []computetensor.SparseIndex,
) (computetensor.SparseTensor, error) {
	_ = shape
	_ = valueDType
	_ = layout
	_ = values
	_ = indices

	return nil, computetensor.ErrLayoutUnsupported
}

func (tensorBackend *TensorBackend) Download(
	input computetensor.Tensor,
) (dtype.DType, []byte, error) {
	if tensorBackend.closed.Load() != 0 {
		return dtype.Invalid, nil, computetensor.ErrBackendClosed
	}

	if input == nil {
		return dtype.Invalid, nil, errors.New("metal tensor: nil input")
	}

	if input.Location() != computetensor.Metal {
		return dtype.Invalid, nil, fmt.Errorf("metal tensor: cannot download %s tensor", input.Location())
	}

	return input.RawBytes()
}

/*
Close releases the backend.
*/
func (tensorBackend *TensorBackend) Close() error {
	tensorBackend.closed.Store(1)
	tensorBackend.cacheMu.Lock()
	defer tensorBackend.cacheMu.Unlock()

	var closeErr error

	for key, entry := range tensorBackend.kvEntries {
		if err := entry.close(); err != nil && closeErr == nil {
			closeErr = err
		}

		delete(tensorBackend.kvEntries, key)
	}

	for key, entry := range tensorBackend.resident {
		if entry.value == nil {
			continue
		}

		if err := entry.value.Close(); err != nil && closeErr == nil {
			closeErr = err
		}

		delete(tensorBackend.resident, key)
	}

	for key, state := range tensorBackend.optimizerStates {
		if err := state.close(); err != nil && closeErr == nil {
			closeErr = err
		}

		delete(tensorBackend.optimizerStates, key)
	}

	if tensorBackend.runtime != nil {
		if err := tensorBackend.runtime.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}

	return closeErr
}

func (tensorBackend *TensorBackend) sharedRuntime(
	previous *MetalRuntime,
) (*MetalRuntime, error) {
	if previous == nil || previous == tensorBackend.runtime {
		return tensorBackend.runtime, nil
	}

	if err := previous.Close(); err != nil {
		return nil, err
	}

	return tensorBackend.runtime, nil
}

func (entry *residentKVEntry) close() error {
	if entry == nil {
		return nil
	}

	var closeErr error

	if entry.key != nil {
		if err := entry.key.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}

	if entry.value != nil {
		if err := entry.value.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}

	entry.key = nil
	entry.value = nil
	entry.shape = nil

	return closeErr
}
