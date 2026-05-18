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
	metalUnaryFloat32Rsqrt
	metalUnaryFloat32Exp
	metalUnaryFloat32Log
	metalUnaryFloat32Sin
	metalUnaryFloat32Cos
	metalUnaryFloat32Tanh
	metalUnaryFloat32Sigmoid
	metalUnaryFloat32Silu
	metalUnaryFloat32Swish
	metalUnaryFloat32Softsign
	metalUnaryFloat32ELU
	metalUnaryFloat32SELU
	metalUnaryFloat32LeakyReLU
	metalUnaryFloat32HardSigmoid
	metalUnaryFloat32HardSwish
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

func runMetalLinear(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = input
	_ = weight
	_ = bias
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalFusedQKV(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
) error {
	_ = input
	_ = weight
	_ = bias
	_ = query
	_ = key
	_ = value

	return tensor.ErrNeedsPlatformSetup
}

func runMetalLoRAMerge(
	baseWeight tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = baseWeight
	_ = loraA
	_ = loraB
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalLoRAApply(
	baseOut tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = baseOut
	_ = loraA
	_ = loraB
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalEmbeddingLookup(
	table tensor.Tensor,
	indices tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = table
	_ = indices
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalEmbeddingBag(
	table tensor.Tensor,
	indices tensor.Tensor,
	offsets tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = table
	_ = indices
	_ = offsets
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalApplyMask(input tensor.Tensor, mask tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = mask
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalCausalMask(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalALiBiBias(scores tensor.Tensor, slope tensor.Tensor, out tensor.Tensor) error {
	_ = scores
	_ = slope
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalConv2D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = input
	_ = weight
	_ = bias
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalConv1D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = input
	_ = weight
	_ = bias
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalConv3D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = input
	_ = weight
	_ = bias
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalConvTranspose2D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = input
	_ = weight
	_ = bias
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalMaxPool2D(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalAvgPool2D(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalAdaptiveAvgPool2D(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalAdaptiveMaxPool2D(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalOptimizer4Kernel(operation metalOptimizerOp, args ...tensor.Tensor) error {
	_ = operation
	_ = args

	return tensor.ErrNeedsPlatformSetup
}

func runMetalOptimizer3Kernel(operation metalOptimizerOp, args ...tensor.Tensor) error {
	_ = operation
	_ = args

	return tensor.ErrNeedsPlatformSetup
}

func runMetalOptimizer2Kernel(operation metalOptimizerOp, args ...tensor.Tensor) error {
	_ = operation
	_ = args

	return tensor.ErrNeedsPlatformSetup
}

func runMetalLARSStep(
	params tensor.Tensor,
	gradients tensor.Tensor,
	momentum tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = params
	_ = gradients
	_ = momentum
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalHebbianStep(
	weights tensor.Tensor,
	post tensor.Tensor,
	pre tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = weights
	_ = post
	_ = pre
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalInt8Dequant(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalInt4Dequant(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalInt8Quant(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalPairLossKernel(operation metalLossOp, args ...tensor.Tensor) error {
	_ = operation
	_ = args

	return tensor.ErrNeedsPlatformSetup
}

func runMetalCrossEntropyLoss(input tensor.Tensor, targets tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = targets
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalReductionKernel(operation metalReductionOp, args ...tensor.Tensor) error {
	_ = operation
	_ = args

	return tensor.ErrNeedsPlatformSetup
}

func runMetalInvSqrtDimScale(input tensor.Tensor, dim tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = dim
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalLogSumExp(input tensor.Tensor, out tensor.Tensor) error {
	_ = input
	_ = out

	return tensor.ErrNeedsPlatformSetup
}

func runMetalOuter(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	_ = left
	_ = right
	_ = out

	return tensor.ErrNeedsPlatformSetup
}
