//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
metalBridge stub for non-darwin or no-cgo builds. Every method
returns ErrNeedsPlatformSetup so callers compile but the device is
clearly unavailable. The darwin+cgo bridge lives in bridge_darwin.go.
*/
type metalBridge struct{}

func openMetalBridge() (*metalBridge, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) recommendedMaxWorkingSet() int64 {
	return 0
}

func (bridge *metalBridge) upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) uploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytesIn []byte,
) (tensor.Tensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) empty(
	shape tensor.Shape,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	_ = shape
	_ = storageDType

	return nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrNeedsPlatformSetup
}

func (bridge *metalBridge) close() error {
	return nil
}

type metalBinaryFloat32Operation int

const (
	metalBinaryFloat32Add metalBinaryFloat32Operation = iota
	metalBinaryFloat32Sub
	metalBinaryFloat32Mul
	metalBinaryFloat32Div
	metalBinaryFloat32Max
	metalBinaryFloat32Min
	metalBinaryFloat32Eq
	metalBinaryFloat32Ne
	metalBinaryFloat32Lt
	metalBinaryFloat32Le
	metalBinaryFloat32Gt
	metalBinaryFloat32Ge
)

func runMetalBinaryFloat32(
	operation metalBinaryFloat32Operation,
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = operation
	_ = left
	_ = right
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalBinaryElementwise(
	operation metalBinaryFloat32Operation,
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = operation
	_ = left
	_ = right
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

type metalUnaryFloat32Operation int

const (
	metalUnaryFloat32Relu metalUnaryFloat32Operation = iota
	metalUnaryFloat32Abs
	metalUnaryFloat32Neg
	metalUnaryFloat32Square
	metalUnaryFloat32Recip
	metalUnaryFloat32Sqrt
	metalUnaryFloat32Sign
)

func runMetalUnaryFloat32(
	operation metalUnaryFloat32Operation,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = operation
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalUnaryElementwise(
	operation metalUnaryFloat32Operation,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = operation
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalReshape(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalMergeHeads(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalSplitHeads(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalViewAsHeads(input tensor.Tensor, heads tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = heads
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalConcat(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	_ = left
	_ = right
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalSplit2(input tensor.Tensor, left tensor.Tensor, right tensor.Tensor) error {
	_ = input
	_ = left
	_ = right

	return tensor.ErrNeedsPlatformSetup
}

func runMetalLastToken(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalTranspose2D(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalUpsampleNearest2D(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}
