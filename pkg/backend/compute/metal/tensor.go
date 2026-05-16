//go:build darwin && cgo

package metal

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ computetensor.Backend = (*TensorBackend)(nil)

/*
TensorBackend owns Metal MTLBuffer tensors.
*/
type TensorBackend struct {
	closed         atomic.Uint32
	cacheMu        sync.Mutex
	runtime        *MetalRuntime
	activationOps  *MetalActivation
	mathOps        *MathOps
	shapeOps       *MetalShapeOps
	attentionOps   *MetalAttention
	positionalOps  *MetalPositional
	convolutionOps *ConvolutionOps
	poolingOps     *PoolingOps
	maskingOps     *MetalMasking
	projectionOps  *ProjectionOps
	vsaOps         *MetalVSAOps
	hawkesOps      *MetalHawkes
	embeddingOps   map[string]*EmbeddingOps
	kvEntries      map[string]*residentKVEntry
	resident       map[string]computetensor.Float64Tensor
}

type residentKVEntry struct {
	epoch    uint64
	capacity int
	shape    []int
	key      computetensor.Float64Tensor
	value    computetensor.Float64Tensor
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
		runtime:      runtime,
		embeddingOps: make(map[string]*EmbeddingOps),
		kvEntries:    make(map[string]*residentKVEntry),
		resident:     make(map[string]computetensor.Float64Tensor),
	}, nil
}

/*
Location identifies Metal storage ownership.
*/
func (tensorBackend *TensorBackend) Location() computetensor.Location {
	return computetensor.Metal
}

/*
UploadFloat64 converts host float64 values into resident Metal float32 storage.
*/
func (tensorBackend *TensorBackend) UploadFloat64(
	shape computetensor.Shape, values []float64,
) (computetensor.Float64Tensor, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("metal tensor: backend is closed")
	}

	if !shape.Valid() {
		return nil, errors.New("metal tensor: invalid shape")
	}

	if shape.Len() != len(values) {
		return nil, fmt.Errorf(
			"metal tensor: shape has %d elements but upload received %d values",
			shape.Len(), len(values),
		)
	}

	return tensorBackend.runtime.UploadFloat64(shape, values)
}

/*
DownloadFloat64 copies resident Metal storage back to host float64 values.
*/
func (tensorBackend *TensorBackend) DownloadFloat64(
	input computetensor.Float64Tensor,
) ([]float64, error) {
	if tensorBackend.closed.Load() != 0 {
		return nil, errors.New("metal tensor: backend is closed")
	}

	if input == nil {
		return nil, errors.New("metal tensor: nil input")
	}

	if input.Location() != computetensor.Metal {
		return nil, fmt.Errorf("metal tensor: cannot download %s tensor", input.Location())
	}

	return input.CloneFloat64()
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

	for key, value := range tensorBackend.resident {
		if value == nil {
			continue
		}

		if err := value.Close(); err != nil && closeErr == nil {
			closeErr = err
		}

		delete(tensorBackend.resident, key)
	}

	if tensorBackend.runtime != nil {
		if err := tensorBackend.runtime.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}

	return closeErr
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
