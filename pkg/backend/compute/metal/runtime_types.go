//go:build darwin && cgo

package metal

import (
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
MetalStorageMode records the Metal resource storage policy used by a resident tensor.
*/
type MetalStorageMode int

const (
	MetalStorageModeShared MetalStorageMode = iota
	MetalStorageModePrivate
)

/*
String returns a stable storage mode label for diagnostics and benchmark reports.
*/
func (storageMode MetalStorageMode) String() string {
	switch storageMode {
	case MetalStorageModeShared:
		return "shared"
	case MetalStorageModePrivate:
		return "private"
	default:
		return "unknown"
	}
}

/*
MetalLayout records the physical layout interpretation for resident storage.
*/
type MetalLayout string

const (
	MetalLayoutContiguous MetalLayout = "contiguous"
)

/*
MetalAllocationKind classifies allocations for runtime metrics.
*/
type MetalAllocationKind string

const (
	MetalAllocationTensor     MetalAllocationKind = "tensor"
	MetalAllocationParameter  MetalAllocationKind = "parameter"
	MetalAllocationScratch    MetalAllocationKind = "scratch"
	MetalAllocationKVCache    MetalAllocationKind = "kv_cache"
	MetalAllocationUpload     MetalAllocationKind = "upload"
	MetalAllocationPersistent MetalAllocationKind = "persistent"
)

/*
MetalTensorMetadata is the dtype-aware physical metadata carried by Metal tensors.
*/
type MetalTensorMetadata struct {
	DType         computetensor.DType
	Shape         computetensor.Shape
	Strides       []int
	ByteSize      int
	StorageMode   MetalStorageMode
	StorageOffset int
	Layout        MetalLayout
	AliasOf       string
	Allocation    MetalAllocationKind
}

/*
MetalRuntimeConfig controls resident tensor allocation policy.
*/
type MetalRuntimeConfig struct {
	DefaultStorageMode MetalStorageMode
	PoolBuffers        bool
}

/*
DefaultMetalRuntimeConfig returns the production storage policy.
*/
func DefaultMetalRuntimeConfig() MetalRuntimeConfig {
	return MetalRuntimeConfig{
		DefaultStorageMode: MetalStorageModePrivate,
		PoolBuffers:        true,
	}
}

/*
MetalRuntimeMetrics reports allocator and transfer counters.
*/
type MetalRuntimeMetrics struct {
	AllocatedBytes int64
	LiveBytes      int64
	PeakBytes      int64
	ReusedBytes    int64
	TransferBytes  int64
	PooledBytes    int64
}

type metalBufferKey struct {
	sizeClass   int
	storageMode MetalStorageMode
}

/*
MetalRuntime owns resident tensor allocation and transfer accounting.
*/
type MetalRuntime struct {
	closed atomic.Uint32
	mu     sync.Mutex
	config MetalRuntimeConfig
	pools  map[metalBufferKey][]unsafe.Pointer
	metric MetalRuntimeMetrics
}

func contiguousStrides(dimensions []int) []int {
	strides := make([]int, len(dimensions))
	stride := 1

	for dimensionIndex := len(dimensions) - 1; dimensionIndex >= 0; dimensionIndex-- {
		strides[dimensionIndex] = stride
		dimension := dimensions[dimensionIndex]

		if dimension == 0 {
			stride = 0

			continue
		}

		stride *= dimension
	}

	return strides
}

func metalSizeClass(bytes int) int {
	if bytes <= 0 {
		return 0
	}

	sizeClass := 4096

	for sizeClass < bytes {
		sizeClass *= 2
	}

	return sizeClass
}

func cloneMetadata(metadata MetalTensorMetadata) MetalTensorMetadata {
	metadata.Strides = slices.Clone(metadata.Strides)

	return metadata
}
