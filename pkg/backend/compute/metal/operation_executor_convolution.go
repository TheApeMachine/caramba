//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalConv1DNodeConfig struct {
	outputShape tensor.Shape
	inputShape  []int
	outputDims  []int
	outChannels int
	kernelSize  int
	stride      int
	padding     int
	dilation    int
	groups      int
}

type metalConv3DNodeConfig struct {
	outputShape    tensor.Shape
	inputShape     []int
	outputDims     []int
	outChannels    int
	kernelDepth    int
	kernelHeight   int
	kernelWidth    int
	strideDepth    int
	strideHeight   int
	strideWidth    int
	padDepth       int
	padHeight      int
	padWidth       int
	dilationDepth  int
	dilationHeight int
	dilationWidth  int
	groups         int
}

func (tensorBackend *TensorBackend) applyConv1D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: conv1d node %q requires 1 input", node.ID)
	}

	config, err := newMetalConv1DNodeConfig(node, inputs[0])
	if err != nil {
		return nil, err
	}

	weightTensor, biasTensor, err := tensorBackend.convolutionTensors(
		node,
		"conv1d",
		config.weightShape(),
		config.biasShape(),
	)
	if err != nil {
		return nil, err
	}

	convolutionOps, err := tensorBackend.convolution()
	if err != nil {
		return nil, err
	}

	return convolutionOps.Conv1dTensor(
		inputs[0],
		weightTensor,
		biasTensor,
		config.outputShape,
		config.inputShape[0],
		config.inputShape[1],
		config.inputShape[2],
		config.outChannels,
		config.kernelSize,
		config.stride,
		config.padding,
		config.dilation,
		config.groups,
	)
}

func (tensorBackend *TensorBackend) applyConv3D(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 1 {
		return nil, fmt.Errorf("metal tensor: conv3d node %q requires 1 input", node.ID)
	}

	config, err := newMetalConv3DNodeConfig(node, inputs[0])
	if err != nil {
		return nil, err
	}

	weightTensor, biasTensor, err := tensorBackend.convolutionTensors(
		node,
		"conv3d",
		config.weightShape(),
		config.biasShape(),
	)
	if err != nil {
		return nil, err
	}

	convolutionOps, err := tensorBackend.convolution()
	if err != nil {
		return nil, err
	}

	return convolutionOps.Conv3dTensor(
		inputs[0],
		weightTensor,
		biasTensor,
		config.outputShape,
		config.inputShape[0],
		config.inputShape[1],
		config.inputShape[2],
		config.inputShape[3],
		config.inputShape[4],
		config.outChannels,
		config.kernelDepth,
		config.kernelHeight,
		config.kernelWidth,
		config.strideDepth,
		config.strideHeight,
		config.strideWidth,
		config.padDepth,
		config.padHeight,
		config.padWidth,
		config.dilationDepth,
		config.dilationHeight,
		config.dilationWidth,
		config.groups,
	)
}

func newMetalConv1DNodeConfig(
	node executor.NodeSpec,
	input tensor.Tensor,
) (metalConv1DNodeConfig, error) {
	outputShape, err := tensor.NewShape(node.Shape)
	if err != nil {
		return metalConv1DNodeConfig{}, err
	}

	outputDims := outputShape.Dims()
	defaultOutChannels := 0

	if len(outputDims) > 1 {
		defaultOutChannels = outputDims[1]
	}

	config := metalConv1DNodeConfig{
		outputShape: outputShape,
		inputShape:  input.Shape().Dims(),
		outputDims:  outputDims,
		kernelSize:  intConfig(node, "kernel_size", 0),
		stride:      intConfig(node, "stride", 1),
		padding:     intConfig(node, "padding", 0),
		dilation:    intConfig(node, "dilation", 1),
		groups:      intConfig(node, "groups", 1),
	}
	config.outChannels = intConfigAny(node, defaultOutChannels, "out_channels", "out_c")
	config.groups = intConfig(node, "num_groups", config.groups)

	return config, config.validate(node.ID)
}

func (config metalConv1DNodeConfig) validate(nodeID string) error {
	if len(config.inputShape) != 3 {
		return fmt.Errorf("metal tensor: conv1d node %q expects NCL rank 3", nodeID)
	}

	if len(config.outputDims) != 3 {
		return fmt.Errorf("metal tensor: conv1d node %q output must be rank 3", nodeID)
	}

	return validateMetalConvNode(
		nodeID,
		"conv1d",
		config.inputShape[1],
		config.outputDims[1],
		config.outChannels,
		config.kernelSize,
		1,
		config.stride,
		1,
		config.dilation,
		1,
		config.groups,
	)
}

func (config metalConv1DNodeConfig) weightShape() []int {
	return []int{
		config.outChannels,
		config.inputShape[1] / config.groups,
		config.kernelSize,
	}
}

func (config metalConv1DNodeConfig) biasShape() []int {
	return []int{config.outChannels}
}

func newMetalConv3DNodeConfig(
	node executor.NodeSpec,
	input tensor.Tensor,
) (metalConv3DNodeConfig, error) {
	outputShape, err := tensor.NewShape(node.Shape)
	if err != nil {
		return metalConv3DNodeConfig{}, err
	}

	outputDims := outputShape.Dims()
	defaultOutChannels := 0

	if len(outputDims) > 1 {
		defaultOutChannels = outputDims[1]
	}

	config := metalConv3DNodeConfig{
		outputShape: outputShape,
		inputShape:  input.Shape().Dims(),
		outputDims:  outputDims,
		outChannels: intConfigAny(
			node, defaultOutChannels, "out_channels", "out_c",
		),
	}
	config.loadKernelConfig(node)
	config.loadStrideConfig(node)
	config.loadPaddingConfig(node)
	config.loadDilationConfig(node)
	config.groups = intConfig(node, "num_groups", intConfig(node, "groups", 1))

	return config, config.validate(node.ID)
}

func (config *metalConv3DNodeConfig) loadKernelConfig(node executor.NodeSpec) {
	kernelSize := intConfig(node, "kernel_size", 0)

	config.kernelDepth = intConfigAny(node, kernelSize, "kernel_d", "k_d")
	config.kernelHeight = intConfigAny(node, kernelSize, "kernel_h", "k_h")
	config.kernelWidth = intConfigAny(node, kernelSize, "kernel_w", "k_w")
}

func (config *metalConv3DNodeConfig) loadStrideConfig(node executor.NodeSpec) {
	stride := intConfig(node, "stride", 1)

	config.strideDepth = intConfigAny(node, stride, "stride_d", "s_d")
	config.strideHeight = intConfigAny(node, stride, "stride_h", "s_h")
	config.strideWidth = intConfigAny(node, stride, "stride_w", "s_w")
}

func (config *metalConv3DNodeConfig) loadPaddingConfig(node executor.NodeSpec) {
	padding := intConfig(node, "padding", 0)

	config.padDepth = intConfigAny(node, padding, "pad_d", "p_d")
	config.padHeight = intConfigAny(node, padding, "pad_h", "p_h")
	config.padWidth = intConfigAny(node, padding, "pad_w", "p_w")
}

func (config *metalConv3DNodeConfig) loadDilationConfig(node executor.NodeSpec) {
	dilation := intConfig(node, "dilation", 1)

	config.dilationDepth = intConfigAny(node, dilation, "dilation_d", "dil_d", "d_d")
	config.dilationHeight = intConfigAny(node, dilation, "dilation_h", "dil_h", "d_h")
	config.dilationWidth = intConfigAny(node, dilation, "dilation_w", "dil_w", "d_w")
}

func (config metalConv3DNodeConfig) validate(nodeID string) error {
	if len(config.inputShape) != 5 {
		return fmt.Errorf("metal tensor: conv3d node %q expects NCDHW rank 5", nodeID)
	}

	if len(config.outputDims) != 5 {
		return fmt.Errorf("metal tensor: conv3d node %q output must be rank 5", nodeID)
	}

	return validateMetalConvNode(
		nodeID,
		"conv3d",
		config.inputShape[1],
		config.outputDims[1],
		config.outChannels,
		config.kernelHeight,
		config.kernelWidth,
		config.strideHeight,
		config.strideWidth,
		config.dilationHeight,
		config.dilationWidth,
		config.groups,
	)
}

func (config metalConv3DNodeConfig) weightShape() []int {
	return []int{
		config.outChannels,
		config.inputShape[1] / config.groups,
		config.kernelDepth,
		config.kernelHeight,
		config.kernelWidth,
	}
}

func (config metalConv3DNodeConfig) biasShape() []int {
	return []int{config.outChannels}
}
