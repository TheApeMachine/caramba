//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) Exp(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.ExpTensor(input)
}

func (tensorBackend *TensorBackend) Sin(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.SinTensor(input)
}

func (tensorBackend *TensorBackend) Cos(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.CosTensor(input)
}

func (tensorBackend *TensorBackend) Log(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.LogTensor(input)
}

func (tensorBackend *TensorBackend) Sign(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.SignTensor(input)
}

func (tensorBackend *TensorBackend) Softmax(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.SoftmaxTensor(input)
}

func (tensorBackend *TensorBackend) LogSumExp(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.LogSumExpTensor(input)
}

func (tensorBackend *TensorBackend) InvSqrtDimScale(
	input tensor.Tensor,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.InvSqrtDimScaleTensor(input)
}

func (tensorBackend *TensorBackend) Dropout(
	input tensor.Tensor,
	probability float64,
	training bool,
	seed int,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.DropoutTensor(input, probability, training, seed)
}

func (tensorBackend *TensorBackend) Outer(
	left tensor.Tensor,
	right tensor.Tensor,
	outputShape tensor.Shape,
) (tensor.Tensor, error) {
	mathOps, err := tensorBackend.math()
	if err != nil {
		return nil, err
	}

	return mathOps.OuterTensor(left, right, outputShape)
}

func (tensorBackend *TensorBackend) applyExp(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.Exp)
}

func (tensorBackend *TensorBackend) applySin(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.Sin)
}

func (tensorBackend *TensorBackend) applyCos(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.Cos)
}

func (tensorBackend *TensorBackend) applyLog(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.Log)
}

func (tensorBackend *TensorBackend) applySign(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.Sign)
}

func (tensorBackend *TensorBackend) applyOuter(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: outer node %q requires 2 inputs", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	return tensorBackend.Outer(inputs[0], inputs[1], outputShape)
}

func (tensorBackend *TensorBackend) applySoftmax(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.Softmax)
}

func (tensorBackend *TensorBackend) applyInvSqrtDimScale(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.InvSqrtDimScale)
}

func (tensorBackend *TensorBackend) applyDropout(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return tensorBackend.applyUnaryMath(
		ctx,
		node,
		inputs,
		func(input tensor.Tensor) (tensor.Tensor, error) {
			return tensorBackend.Dropout(
				input,
				floatConfig(node, "p", 0),
				boolConfig(node, "training", false),
				intConfigAny(node, 0, "seed", "step"),
			)
		},
	)
}

func (tensorBackend *TensorBackend) applyLogSumExp(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	output, err := tensorBackend.applyUnaryMath(ctx, node, inputs, tensorBackend.LogSumExp)
	if err != nil {
		return nil, err
	}

	return requireNodeOutputShape(node, output)
}

func (tensorBackend *TensorBackend) applyUnaryMath(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
	apply func(tensor.Tensor) (tensor.Tensor, error),
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: %s node %q requires 1 input", node.Op, node.ID)
	}

	return apply(inputs[0])
}

func requireNodeOutputShape(
	node executor.NodeSpec,
	output tensor.Tensor,
) (tensor.Tensor, error) {
	expected, err := tensor.NewShape(node.Shape)
	if err != nil {
		_ = output.Close()

		return nil, err
	}

	if expected.Equal(output.Shape()) {
		return output, nil
	}

	_ = output.Close()

	return nil, fmt.Errorf(
		"metal tensor: %s node %q output shape %v does not match %v",
		node.Op,
		node.ID,
		output.Shape().Dims(),
		expected.Dims(),
	)
}
