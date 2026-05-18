//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyMaxPool2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	outputShape, poolingOps, err := tensorBackend.poolingInputs(ctx, node, inputs)
	if err != nil {
		return nil, err
	}

	return poolingOps.MaxPool2dTensor(inputs[0], outputShape, maxPoolNodeParams(node))
}

func (tensorBackend *TensorBackend) applyAvgPool2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	outputShape, poolingOps, err := tensorBackend.poolingInputs(ctx, node, inputs)
	if err != nil {
		return nil, err
	}

	return poolingOps.AvgPool2dTensor(inputs[0], outputShape, avgPoolNodeParams(node))
}

func (tensorBackend *TensorBackend) applyAdaptiveAvgPool2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	outputShape, poolingOps, err := tensorBackend.poolingInputs(ctx, node, inputs)
	if err != nil {
		return nil, err
	}

	return poolingOps.AdaptiveAvgPool2dTensor(inputs[0], outputShape)
}

func (tensorBackend *TensorBackend) applyAdaptiveMaxPool2D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	outputShape, poolingOps, err := tensorBackend.poolingInputs(ctx, node, inputs)
	if err != nil {
		return nil, err
	}

	return poolingOps.AdaptiveMaxPool2dTensor(inputs[0], outputShape)
}

func (tensorBackend *TensorBackend) poolingInputs(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Shape, *PoolingOps, error) {
	if err := ctx.Err(); err != nil {
		return tensor.Shape{}, nil, err
	}

	if len(inputs) != 1 {
		return tensor.Shape{}, nil, fmt.Errorf(
			"metal tensor: %s node %q requires 1 input",
			node.Op,
			node.ID,
		)
	}

	outputShape, err := tensor.NewShape(node.Shape)
	if err != nil {
		return tensor.Shape{}, nil, err
	}

	poolingOps, err := tensorBackend.pooling()
	if err != nil {
		return tensor.Shape{}, nil, err
	}

	return outputShape, poolingOps, nil
}

func maxPoolNodeParams(node executor.NodeSpec) MaxPool2dParams {
	kernelSize := intConfig(node, "kernel_size", 0)
	stride := intConfig(node, "stride", 1)
	padding := intConfig(node, "padding", 0)
	dilation := intConfig(node, "dilation", 1)

	return MaxPool2dParams{
		KernelH:   intConfigAny(node, kernelSize, "kernel_h", "k_h"),
		KernelW:   intConfigAny(node, kernelSize, "kernel_w", "k_w"),
		StrideH:   intConfigAny(node, stride, "stride_h", "s_h"),
		StrideW:   intConfigAny(node, stride, "stride_w", "s_w"),
		PadH:      intConfigAny(node, padding, "pad_h", "p_h"),
		PadW:      intConfigAny(node, padding, "pad_w", "p_w"),
		DilationH: intConfigAny(node, dilation, "dilation_h", "d_h"),
		DilationW: intConfigAny(node, dilation, "dilation_w", "d_w"),
		CeilMode:  boolConfig(node, "ceil", false),
	}
}

func avgPoolNodeParams(node executor.NodeSpec) AvgPool2dParams {
	maxParams := maxPoolNodeParams(node)

	return AvgPool2dParams{
		KernelH:         maxParams.KernelH,
		KernelW:         maxParams.KernelW,
		StrideH:         maxParams.StrideH,
		StrideW:         maxParams.StrideW,
		PadH:            maxParams.PadH,
		PadW:            maxParams.PadW,
		DilationH:       maxParams.DilationH,
		DilationW:       maxParams.DilationW,
		CeilMode:        maxParams.CeilMode,
		CountIncludePad: boolConfig(node, "count_include_pad", boolConfig(node, "count_pad", false)),
		DivisorOverride: intConfig(node, "divisor_override", 0),
	}
}
