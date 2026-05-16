//go:build darwin && cgo

package metal

// #include "tensor.h"
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
NewMetalRuntime creates the process-local Metal runtime handle used by graph execution.
*/
func NewMetalRuntime(config MetalRuntimeConfig) (*MetalRuntime, error) {
	if config.DefaultStorageMode != MetalStorageModeShared &&
		config.DefaultStorageMode != MetalStorageModePrivate {
		return nil, fmt.Errorf("metal runtime: unsupported storage mode %d", config.DefaultStorageMode)
	}

	if rc := C.metal_tensor_init(); rc != 0 {
		return nil, fmt.Errorf("metal runtime: initialization failed (rc=%d)", rc)
	}

	return &MetalRuntime{
		config: config,
		pools:  make(map[metalBufferKey][]unsafe.Pointer),
	}, nil
}

func newStandaloneMetalRuntime() (*MetalRuntime, error) {
	config := DefaultMetalRuntimeConfig()
	config.PoolBuffers = false

	return NewMetalRuntime(config)
}

/*
NewFloat32Tensor allocates resident float32 storage for an operation output.
*/
func (runtime *MetalRuntime) NewFloat32Tensor(
	shape computetensor.Shape,
	allocation MetalAllocationKind,
) (*Tensor, error) {
	return runtime.newTensor(shape, computetensor.Float32, runtime.config.DefaultStorageMode, allocation)
}

/*
UploadFloat64 uploads host values into dtype-aware resident float32 storage.
*/
func (runtime *MetalRuntime) UploadFloat64(
	shape computetensor.Shape,
	values []float64,
) (*Tensor, error) {
	if runtime.closed.Load() != 0 {
		return nil, errors.New("metal runtime: runtime is closed")
	}

	if !shape.Valid() {
		return nil, errors.New("metal runtime: invalid upload shape")
	}

	if shape.Len() != len(values) {
		return nil, fmt.Errorf(
			"metal runtime: shape has %d elements but upload received %d values",
			shape.Len(), len(values),
		)
	}

	bytes, err := shape.Bytes(computetensor.Float32)
	if err != nil {
		return nil, err
	}

	float32Values := toFloat32(values)
	var buffer unsafe.Pointer

	if len(float32Values) > 0 {
		buffer = C.metal_tensor_upload_float32_mode(
			(*C.float)(unsafe.Pointer(&float32Values[0])),
			C.size_t(len(float32Values)),
			C.int(runtime.config.DefaultStorageMode),
		)

		if buffer == nil {
			return nil, fmt.Errorf("metal runtime: upload failed")
		}
	}

	runtime.recordAllocate(bytes, bytes)
	runtime.recordTransfer(bytes)

	return runtime.wrapTensor(
		shape,
		computetensor.Float32,
		runtime.config.DefaultStorageMode,
		buffer,
		bytes,
		MetalAllocationUpload,
	), nil
}

/*
Metrics returns a snapshot of runtime allocator counters.
*/
func (runtime *MetalRuntime) Metrics() MetalRuntimeMetrics {
	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	return runtime.metric
}

/*
Close releases pooled buffers owned by the runtime.
*/
func (runtime *MetalRuntime) Close() error {
	if runtime == nil {
		return nil
	}

	if !runtime.closed.CompareAndSwap(0, 1) {
		return nil
	}

	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	var closeErr error

	for key, buffers := range runtime.pools {
		for _, buffer := range buffers {
			if rc := C.metal_tensor_free(buffer); rc != 0 && closeErr == nil {
				closeErr = fmt.Errorf("metal runtime: free failed (rc=%d)", rc)
			}
		}

		delete(runtime.pools, key)
	}

	runtime.metric.PooledBytes = 0

	return closeErr
}

func (runtime *MetalRuntime) newTensor(
	shape computetensor.Shape,
	dtype computetensor.DType,
	storageMode MetalStorageMode,
	allocation MetalAllocationKind,
) (*Tensor, error) {
	if runtime.closed.Load() != 0 {
		return nil, errors.New("metal runtime: runtime is closed")
	}

	if !shape.Valid() {
		return nil, errors.New("metal runtime: invalid shape")
	}

	if dtype != computetensor.Float32 {
		return nil, fmt.Errorf("metal runtime: unsupported resident dtype %q", dtype)
	}

	bytes, err := shape.Bytes(dtype)
	if err != nil {
		return nil, err
	}

	var buffer unsafe.Pointer
	actualBytes := bytes

	if shape.Len() > 0 {
		buffer, actualBytes = runtime.acquireBuffer(bytes, storageMode)

		if buffer == nil {
			actualBytes = bytes
			if runtime.config.PoolBuffers {
				actualBytes = metalSizeClass(bytes)
			}

			elementCount := actualBytes / mustDTypeSize(dtype)
			buffer = C.metal_tensor_empty_float32_mode(
				C.size_t(elementCount),
				C.int(storageMode),
			)
		}

		if buffer == nil {
			return nil, fmt.Errorf("metal runtime: allocation of %d bytes failed", bytes)
		}
	}

	runtime.recordAllocate(bytes, actualBytes)

	return runtime.wrapTensor(shape, dtype, storageMode, buffer, bytes, allocation), nil
}

func mustDTypeSize(dtype computetensor.DType) int {
	size, err := dtype.Size()

	if err != nil {
		panic(err)
	}

	return size
}

func (runtime *MetalRuntime) wrapTensor(
	shape computetensor.Shape,
	dtype computetensor.DType,
	storageMode MetalStorageMode,
	buffer unsafe.Pointer,
	bytes int,
	allocation MetalAllocationKind,
) *Tensor {
	return &Tensor{
		bytes:   bytes,
		shape:   shape,
		buffer:  buffer,
		runtime: runtime,
		metadata: MetalTensorMetadata{
			DType:       dtype,
			Shape:       shape,
			Strides:     contiguousStrides(shape.Dims()),
			ByteSize:    bytes,
			StorageMode: storageMode,
			Layout:      MetalLayoutContiguous,
			Allocation:  allocation,
		},
	}
}

func (runtime *MetalRuntime) release(tensor *Tensor) error {
	if tensor == nil || tensor.buffer == nil {
		return nil
	}

	buffer := tensor.buffer
	bytes := tensor.bytes
	storageMode := tensor.metadata.StorageMode
	tensor.buffer = nil
	tensor.bytes = 0

	runtime.recordRelease(bytes)

	if runtime.closed.Load() != 0 || !runtime.config.PoolBuffers {
		if rc := C.metal_tensor_free(buffer); rc != 0 {
			return fmt.Errorf("metal runtime: free failed (rc=%d)", rc)
		}

		return nil
	}

	sizeClass := metalSizeClass(bytes)
	key := metalBufferKey{sizeClass: sizeClass, storageMode: storageMode}

	runtime.mu.Lock()
	runtime.pools[key] = append(runtime.pools[key], buffer)
	runtime.metric.PooledBytes += int64(sizeClass)
	runtime.mu.Unlock()

	return nil
}

func (runtime *MetalRuntime) acquireBuffer(
	bytes int,
	storageMode MetalStorageMode,
) (unsafe.Pointer, int) {
	if !runtime.config.PoolBuffers {
		return nil, 0
	}

	sizeClass := metalSizeClass(bytes)
	key := metalBufferKey{sizeClass: sizeClass, storageMode: storageMode}

	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	buffers := runtime.pools[key]
	if len(buffers) == 0 {
		return nil, 0
	}

	lastIndex := len(buffers) - 1
	buffer := buffers[lastIndex]
	runtime.pools[key] = buffers[:lastIndex]
	runtime.metric.PooledBytes -= int64(sizeClass)
	runtime.metric.ReusedBytes += int64(bytes)

	return buffer, sizeClass
}

func (runtime *MetalRuntime) recordAllocate(bytes, actualBytes int) {
	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	runtime.metric.AllocatedBytes += int64(actualBytes)
	runtime.metric.LiveBytes += int64(bytes)

	if runtime.metric.LiveBytes > runtime.metric.PeakBytes {
		runtime.metric.PeakBytes = runtime.metric.LiveBytes
	}
}

func (runtime *MetalRuntime) recordRelease(bytes int) {
	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	runtime.metric.LiveBytes -= int64(bytes)
	if runtime.metric.LiveBytes < 0 {
		runtime.metric.LiveBytes = 0
	}
}

func (runtime *MetalRuntime) recordTransfer(bytes int) {
	runtime.mu.Lock()
	defer runtime.mu.Unlock()

	runtime.metric.TransferBytes += int64(bytes)
}
