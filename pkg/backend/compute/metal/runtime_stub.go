//go:build !darwin || !cgo

package metal

import (
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type MetalStorageMode int

const (
	MetalStorageModeShared MetalStorageMode = iota
	MetalStorageModePrivate
)

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

type MetalLayout string

const (
	MetalLayoutContiguous MetalLayout = "contiguous"
)

type MetalAllocationKind string

const (
	MetalAllocationTensor     MetalAllocationKind = "tensor"
	MetalAllocationParameter  MetalAllocationKind = "parameter"
	MetalAllocationScratch    MetalAllocationKind = "scratch"
	MetalAllocationKVCache    MetalAllocationKind = "kv_cache"
	MetalAllocationUpload     MetalAllocationKind = "upload"
	MetalAllocationPersistent MetalAllocationKind = "persistent"
)

type MetalTensorMetadata struct {
	DType         dtype.DType
	Shape         computetensor.Shape
	Strides       []int
	ByteSize      int
	StorageMode   MetalStorageMode
	StorageOffset int
	Layout        MetalLayout
	AliasOf       string
	Allocation    MetalAllocationKind
}

type MetalRuntimeConfig struct {
	DefaultStorageMode MetalStorageMode
	PoolBuffers        bool
}

func DefaultMetalRuntimeConfig() MetalRuntimeConfig {
	return MetalRuntimeConfig{
		DefaultStorageMode: MetalStorageModePrivate,
		PoolBuffers:        true,
	}
}

type MetalRuntimeMetrics struct {
	AllocatedBytes int64
	LiveBytes      int64
	PeakBytes      int64
	ReusedBytes    int64
	TransferBytes  int64
	PooledBytes    int64
}

type MetalRuntime struct{}

func NewMetalRuntime(config MetalRuntimeConfig) (*MetalRuntime, error) {
	return nil, metalUnavailable()
}

func newStandaloneMetalRuntime() (*MetalRuntime, error) {
	return nil, metalUnavailable()
}

func (*MetalRuntime) NewFloat32Tensor(
	shape computetensor.Shape,
	allocation MetalAllocationKind,
) (*Tensor, error) {
	return nil, metalUnavailable()
}

func (*MetalRuntime) Upload(
	shape computetensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (*Tensor, error) {
	_ = sourceDType
	_ = bytes

	return nil, metalUnavailable()
}

func (*MetalRuntime) Metrics() MetalRuntimeMetrics {
	return MetalRuntimeMetrics{}
}

func (*MetalRuntime) Close() error {
	return nil
}
