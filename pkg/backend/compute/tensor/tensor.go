package tensor

import (
	"errors"
	"fmt"
	"slices"
	"sync"
)

const maxInt = int(^uint(0) >> 1)

const hostArenaFloat64Elements = 8 * 1024 * 1024

var (
	errClosedBackend = errors.New("tensor: backend is closed")
	errClosedTensor  = errors.New("tensor: tensor is closed")
	errInvalidShape  = errors.New("tensor: shape is invalid")
	errLiveTensors   = errors.New("tensor: host arena still has live tensors")
)

/*
DType identifies the scalar representation stored by a tensor.
*/
type DType string

const (
	Float64 DType = "float64"
	Float32 DType = "float32"
	Int64   DType = "int64"
)

/*
Size returns the number of bytes used by one scalar value.
*/
func (dtype DType) Size() (int, error) {
	switch dtype {
	case Float64, Int64:
		return 8, nil
	case Float32:
		return 4, nil
	default:
		return 0, fmt.Errorf("tensor: unsupported dtype %q", dtype)
	}
}

/*
Location identifies where tensor storage is owned.
*/
type Location string

const (
	Host    Location = "host"
	CUDA    Location = "cuda"
	Metal   Location = "metal"
	XLA     Location = "xla"
	Network Location = "network"
)

/*
Shape is validated tensor metadata with a cached element count.
*/
type Shape struct {
	dims     []int
	elements int
	valid    bool
}

/*
NewShape validates dimensions and records the element count once.
*/
func NewShape(dims []int) (Shape, error) {
	shape := Shape{
		dims:     slices.Clone(dims),
		elements: 1,
		valid:    true,
	}

	for dimensionIndex, dimension := range shape.dims {
		if dimension < 0 {
			return Shape{}, fmt.Errorf(
				"%w: dimension %d is negative (%d)",
				errInvalidShape, dimensionIndex, dimension,
			)
		}

		if dimension == 0 {
			shape.elements = 0
			continue
		}

		if shape.elements > maxInt/dimension {
			return Shape{}, fmt.Errorf("%w: element count overflows int", errInvalidShape)
		}

		shape.elements *= dimension
	}

	return shape, nil
}

/*
Valid reports whether the shape came from NewShape.
*/
func (shape Shape) Valid() bool {
	return shape.valid
}

/*
Dims returns an immutable copy of the dimensions.
*/
func (shape Shape) Dims() []int {
	return slices.Clone(shape.dims)
}

/*
Len returns the number of elements addressed by the shape.
*/
func (shape Shape) Len() int {
	return shape.elements
}

/*
Bytes returns the storage footprint for this shape and dtype.
*/
func (shape Shape) Bytes(dtype DType) (int, error) {
	if !shape.valid {
		return 0, errInvalidShape
	}

	size, err := dtype.Size()

	if err != nil {
		return 0, err
	}

	if shape.elements > maxInt/size {
		return 0, fmt.Errorf("%w: byte count overflows int", errInvalidShape)
	}

	return shape.elements * size, nil
}

/*
Equal reports whether two shapes address the same dimensions.
*/
func (shape Shape) Equal(other Shape) bool {
	return slices.Equal(shape.dims, other.dims) && shape.elements == other.elements && shape.valid == other.valid
}

/*
Tensor is the backend-neutral contract for persistent compute storage.
*/
type Tensor interface {
	Shape() Shape
	DType() DType
	Location() Location
	Len() int
	Bytes() int
	Close() error
}

/*
Float64Tensor is a tensor that can be copied back to host float64 values.
*/
type Float64Tensor interface {
	Tensor
	CloneFloat64() ([]float64, error)
}

/*
Float64From creates a host-owned float64 tensor from a one-dimensional value slice.
*/
func Float64From(values []float64) (Float64Tensor, error) {
	shape, err := NewShape([]int{len(values)})

	if err != nil {
		return nil, err
	}

	return NewHostBackend().UploadFloat64(shape, values)
}

/*
MustFloat64From is the panic-on-error variant of Float64From for compact setup.
*/
func MustFloat64From(values []float64) Float64Tensor {
	value, err := Float64From(values)

	if err != nil {
		panic(err)
	}

	return value
}

/*
MustCloneFloat64 is the panic-on-error variant of Float64Tensor.CloneFloat64.
*/
func MustCloneFloat64(value Float64Tensor) []float64 {
	values, err := value.CloneFloat64()

	if err != nil {
		panic(err)
	}

	return values
}

/*
Backend owns persistent tensors for one compute location.
*/
type Backend interface {
	Location() Location
	UploadFloat64(shape Shape, values []float64) (Float64Tensor, error)
	DownloadFloat64(tensor Float64Tensor) ([]float64, error)
	Close() error
}

/*
HostBackend owns tensors backed by Go heap storage using a linear memory arena.
The arena is fixed at 8,388,608 float64 elements (64 MiB); UploadFloat64 returns
an error when that arena is exhausted instead of silently allocating elsewhere.
*/
type HostBackend struct {
	mu     sync.RWMutex
	closed bool
	arena  []float64
	offset int
	live   int
}

/*
NewHostBackend creates a backend for native Go tensor ownership.
It allocates a fixed 64 MiB arena to keep allocation behavior explicit.
*/
func NewHostBackend() *HostBackend {
	return &HostBackend{
		arena: make([]float64, hostArenaFloat64Elements),
	}
}

/*
Location identifies this backend as host-owned storage.
*/
func (hostBackend *HostBackend) Location() Location {
	return Host
}

/*
UploadFloat64 copies host values into a persistent host tensor.
*/
func (hostBackend *HostBackend) UploadFloat64(
	shape Shape, values []float64,
) (Float64Tensor, error) {
	return hostBackend.createFloat64(shape, values, true)
}

/*
AdoptFloat64 forwards values to createFloat64 with copy=false: it takes ownership of the
provided slice without copying. The caller must not retain the backing array or mutate it
after AdoptFloat64 returns, or tensor storage will be corrupted. Use UploadFloat64 for a
copied buffer, or CloneFloat64 on the returned tensor when an independent host slice is needed.
*/
func (hostBackend *HostBackend) AdoptFloat64(
	shape Shape, values []float64,
) (Float64Tensor, error) {
	return hostBackend.createFloat64(shape, values, false)
}

func (hostBackend *HostBackend) createFloat64(
	shape Shape, values []float64, copyValues bool,
) (Float64Tensor, error) {
	hostBackend.mu.Lock()
	defer hostBackend.mu.Unlock()

	if hostBackend.closed {
		return nil, errClosedBackend
	}

	if !shape.Valid() {
		return nil, errInvalidShape
	}

	if shape.Len() != len(values) {
		return nil, fmt.Errorf(
			"tensor: shape has %d elements but upload received %d float64 values",
			shape.Len(), len(values),
		)
	}

	bytes, err := shape.Bytes(Float64)

	if err != nil {
		return nil, err
	}

	var dest []float64

	if copyValues {
		if shape.Len() == 0 {
			return &HostTensor{bytes: bytes, shape: shape, values: nil}, nil
		}

		if shape.Len() > len(hostBackend.arena)-hostBackend.offset {
			return nil, fmt.Errorf(
				"tensor: host arena exhausted: requested %d float64 elements with %d remaining of %d",
				shape.Len(), len(hostBackend.arena)-hostBackend.offset, len(hostBackend.arena),
			)
		}

		dest = hostBackend.arena[hostBackend.offset : hostBackend.offset+shape.Len()]
		hostBackend.offset += shape.Len()
		copy(dest, values)
		hostBackend.live++
	} else {
		dest = values
	}

	return &HostTensor{
		backend: hostBackend,
		arena:   copyValues && shape.Len() > 0,
		bytes:   bytes,
		shape:   shape,
		values:  dest,
	}, nil
}

/*
DownloadFloat64 returns a zero-copy host view for tensors stored on Host.

For *HostTensor this aliases the backing slice (not a clone). Callers that need
an independent buffer must use Float64Tensor.CloneFloat64 on the tensor first.
*/
func (hostBackend *HostBackend) DownloadFloat64(tensor Float64Tensor) ([]float64, error) {
	hostBackend.mu.RLock()
	closed := hostBackend.closed
	hostBackend.mu.RUnlock()

	if closed {
		return nil, errClosedBackend
	}

	if tensor == nil {
		return nil, errors.New("tensor: cannot download nil tensor")
	}

	if tensor.Location() != Host {
		return nil, fmt.Errorf("tensor: host backend cannot download %s tensor", tensor.Location())
	}

	hostTensor, ok := tensor.(*HostTensor)

	if !ok {
		return nil, fmt.Errorf("tensor: host backend expected *HostTensor, got %T", tensor)
	}

	return hostTensor.Float64()
}

/*
Close releases the backend. Existing tensors remain responsible for themselves.
*/
func (hostBackend *HostBackend) Close() error {
	hostBackend.mu.Lock()
	defer hostBackend.mu.Unlock()

	hostBackend.closed = true
	hostBackend.arena = nil

	return nil
}

/*
Reset performs a one-shot reclamation of the memory arena for when all tensors are gone.
*/
func (hostBackend *HostBackend) Reset() error {
	hostBackend.mu.Lock()
	defer hostBackend.mu.Unlock()

	if hostBackend.closed {
		return errClosedBackend
	}

	if hostBackend.live != 0 {
		return fmt.Errorf("%w: %d", errLiveTensors, hostBackend.live)
	}

	hostBackend.offset = 0

	return nil
}

func (hostBackend *HostBackend) releaseArenaTensor() {
	hostBackend.mu.Lock()
	defer hostBackend.mu.Unlock()

	if hostBackend.live > 0 {
		hostBackend.live--
	}
}

/*
HostTensor is persistent tensor storage backed by a Go float64 slice.
*/
type HostTensor struct {
	mu      sync.RWMutex
	backend *HostBackend
	arena   bool
	bytes   int
	shape   Shape
	values  []float64
	closed  bool
}

/*
Shape returns validated tensor dimensions.
*/
func (hostTensor *HostTensor) Shape() Shape {
	hostTensor.mu.RLock()
	defer hostTensor.mu.RUnlock()

	return hostTensor.shape
}

/*
DType reports the scalar representation.
*/
func (hostTensor *HostTensor) DType() DType {
	return Float64
}

/*
Location reports host storage ownership.
*/
func (hostTensor *HostTensor) Location() Location {
	return Host
}

/*
Len returns the number of tensor elements.
*/
func (hostTensor *HostTensor) Len() int {
	hostTensor.mu.RLock()
	defer hostTensor.mu.RUnlock()

	return hostTensor.shape.Len()
}

/*
Bytes returns the number of bytes owned by this tensor.
*/
func (hostTensor *HostTensor) Bytes() int {
	hostTensor.mu.RLock()
	defer hostTensor.mu.RUnlock()

	return hostTensor.bytes
}

/*
Float64 returns the zero-copy host view for CPU kernels.

The returned slice aliases HostTensor.values: mutating elements changes the
tensor in place. Copy the slice or use CloneFloat64 when an independent buffer is required.
*/
func (hostTensor *HostTensor) Float64() ([]float64, error) {
	hostTensor.mu.RLock()
	defer hostTensor.mu.RUnlock()

	if hostTensor.closed {
		return nil, errClosedTensor
	}

	return hostTensor.values, nil
}

/*
CloneFloat64 returns a host copy suitable for crossing backend boundaries.
*/
func (hostTensor *HostTensor) CloneFloat64() ([]float64, error) {
	hostTensor.mu.RLock()
	defer hostTensor.mu.RUnlock()

	if hostTensor.closed {
		return nil, errClosedTensor
	}

	return slices.Clone(hostTensor.values), nil
}

/*
Close releases host tensor storage. Idempotent: repeated Close calls are safe.
*/
func (hostTensor *HostTensor) Close() error {
	hostTensor.mu.Lock()
	defer hostTensor.mu.Unlock()

	if hostTensor.closed {
		return nil
	}

	hostTensor.closed = true
	backend := hostTensor.backend
	arena := hostTensor.arena
	hostTensor.backend = nil
	hostTensor.arena = false
	hostTensor.bytes = 0
	hostTensor.values = nil
	hostTensor.shape = Shape{}

	if arena && backend != nil {
		backend.releaseArenaTensor()
	}

	return nil
}
