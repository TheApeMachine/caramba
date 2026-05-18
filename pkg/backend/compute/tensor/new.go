package tensor

import "github.com/theapemachine/caramba/pkg/dtype"

/*
New returns a host tensor with uninitialized storage of the given
shape and dtype. Storage is drawn from the tiered allocator and is
NOT zeroed; callers that need zeroed memory call NewZeroed instead.

This is the canonical entry point for creating host tensors outside
of a Backend.Upload call (e.g. for output buffers of kernels).
*/
func New(shape Shape, asType dtype.DType) (Tensor, error) {
	if !shape.Valid() {
		return nil, ErrShapeInvalid
	}

	bytesNeeded, err := shape.Bytes(asType)

	if err != nil {
		return nil, err
	}

	buffer := Allocate(bytesNeeded)

	return newHostTensor(nil, shape, asType, buffer), nil
}

/*
NewZeroed returns a host tensor with explicitly zeroed storage of
the given shape and dtype.
*/
func NewZeroed(shape Shape, asType dtype.DType) (Tensor, error) {
	tensor, err := New(shape, asType)

	if err != nil {
		return nil, err
	}

	host, ok := tensor.(*HostTensor)

	if !ok {
		return tensor, nil
	}

	for index := range host.bytes {
		host.bytes[index] = 0
	}

	return host, nil
}
