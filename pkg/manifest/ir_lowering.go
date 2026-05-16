package manifest

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func LowerGraphToIR(graph *Graph, defaultShape tensor.Shape) (*ir.Graph, error) {
	return LowerGraphToIRWithInputShapes(graph, defaultShape, nil)
}

func LowerGraphToIRWithInputShapes(
	graph *Graph,
	defaultShape tensor.Shape,
	inputShapeOverrides map[string]tensor.Shape,
) (*ir.Graph, error) {
	if graph == nil {
		return nil, fmt.Errorf("manifest: nil graph")
	}

	irGraph := ir.NewGraph()
	externalInputs := graph.ExternalInputs()
	nodes := make(map[string]*ir.Node, len(graph.nodes)+len(externalInputs))
	bindingShapes := make(map[string]tensor.Shape, len(graph.nodes)+len(externalInputs))
	producedBindings := make(map[string]string, len(graph.nodes))

	for _, manifestNode := range graph.Nodes() {
		producedBindings[publicationBinding(manifestNode)] = manifestNode.ID
	}

	for _, input := range externalInputs {
		if producerID := producedBindings[input]; producerID != "" {
			return nil, fmt.Errorf(
				"manifest: external input %q conflicts with producer node %q",
				input, producerID,
			)
		}

		inputShape, err := externalInputShape(input, defaultShape, inputShapeOverrides)

		if err != nil {
			return nil, err
		}

		node := ir.NewNode(input, ir.OpInput, inputShape)
		node.SetOperationID(ir.OpID("data.input"))
		node.SetMetadata("binding", input)
		node.SetAttribute("out.0", ir.StringAttribute(input))

		nodes[input] = node
		bindingShapes[input] = inputShape
		irGraph.AddNode(node)
	}

	for _, manifestNode := range graph.Nodes() {
		if nodes[manifestNode.ID] != nil {
			return nil, fmt.Errorf("manifest: node %q conflicts with external input", manifestNode.ID)
		}

		opID := manifestNode.OpID
		if opID == "" {
			opID = manifestNode.ID
		}

		inputShapes, err := inputShapesForNode(manifestNode, bindingShapes, defaultShape)

		if err != nil {
			return nil, err
		}

		opShape := operationShapeForNode(inputShapes, defaultShape)
		outputShape, err := outputShapeForNode(manifestNode, opShape, inputShapes)

		if err != nil {
			return nil, err
		}

		node := ir.NewNode(manifestNode.ID, ir.OpType(opID), outputShape)
		node.SetOperationID(ir.OpID(opID))
		node.SetMetadata("op_shape", opShape.Dims())

		for key, value := range manifestNode.Config {
			node.SetMetadata(key, value)
			node.SetAttribute(key, attributeFromValue(value))
		}

		if precision, ok := manifestNode.Config["precision"].(string); ok {
			node.SetValueType(ir.ValueType{
				Shape:     outputShape,
				DType:     tensor.Float64,
				Precision: tensor.DType(precision),
			})
		}

		for index, port := range manifestNode.In {
			node.SetAttribute(fmt.Sprintf("in.%d", index), ir.StringAttribute(port))
		}

		for index, port := range manifestNode.Out {
			node.SetAttribute(fmt.Sprintf("out.%d", index), ir.StringAttribute(port))
		}

		nodes[manifestNode.ID] = node
		bindingShapes[publicationBinding(manifestNode)] = outputShape
		irGraph.AddNode(node)
	}

	for _, manifestNode := range graph.Nodes() {
		node := nodes[manifestNode.ID]

		for _, inputBinding := range manifestNode.In {
			if !graph.externalInputs[inputBinding] {
				continue
			}

			inputNode := nodes[inputBinding]

			if inputNode == nil {
				return nil, fmt.Errorf(
					"manifest: declared external input %q has no IR node",
					inputBinding,
				)
			}

			node.AddInput(inputNode)
		}
	}

	for _, edge := range graph.Edges() {
		to := nodes[edge.To]
		from := nodes[edge.From]

		if to == nil || from == nil {
			return nil, fmt.Errorf("manifest: edge %s -> %s references missing node", edge.From, edge.To)
		}

		to.AddInput(from)
	}

	return irGraph, irGraph.Verify()
}

func externalInputShape(
	input string,
	defaultShape tensor.Shape,
	inputShapeOverrides map[string]tensor.Shape,
) (tensor.Shape, error) {
	shape, ok := inputShapeOverrides[input]

	if !ok {
		return defaultShape, nil
	}

	if !shape.Valid() {
		return tensor.Shape{}, fmt.Errorf("manifest: external input %q has invalid shape", input)
	}

	return shape, nil
}

func inputShapesForNode(
	node *Node, bindingShapes map[string]tensor.Shape, fallback tensor.Shape,
) ([]tensor.Shape, error) {
	if len(node.In) == 0 {
		return []tensor.Shape{fallback}, nil
	}

	shapes := make([]tensor.Shape, len(node.In))

	for index, input := range node.In {
		shape := bindingShapes[input]

		if !shape.Valid() {
			return nil, fmt.Errorf("manifest: node %q input %q has no lowered shape", node.ID, input)
		}

		shapes[index] = shape
	}

	return shapes, nil
}

func operationShapeForNode(inputShapes []tensor.Shape, fallback tensor.Shape) tensor.Shape {
	if len(inputShapes) == 0 || !inputShapes[0].Valid() {
		return fallback
	}

	return inputShapes[0]
}

func outputShapeForNode(
	node *Node,
	opShape tensor.Shape,
	inputShapes []tensor.Shape,
) (tensor.Shape, error) {
	dimensions := opShape.Dims()

	switch node.OpID {
	case "embedding.token":
		return appendShapeDim(dimensions, configInt(node.Config, "d_model", 0))
	case "projection.linear":
		return replaceLastShapeDim(dimensions, configInt(node.Config, "out_features", 0))
	case "projection.fused_qkv":
		out := configInt(node.Config, "d_q", 0) +
			configInt(node.Config, "d_k", 0) +
			configInt(node.Config, "d_v", 0)

		return replaceLastShapeDim(dimensions, out)
	case "activation.swiglu":
		if len(dimensions) == 0 {
			return tensor.NewShape([]int{opShape.Len() / 2})
		}

		out := append([]int(nil), dimensions...)
		last := len(out) - 1
		out[last] /= 2

		return tensor.NewShape(out)
	case "shape.view_as_heads":
		return viewAsHeadsShape(dimensions, configInt(node.Config, "num_heads", 0))
	case "shape.last_token":
		return lastTokenShape(dimensions)
	case "shape.merge_heads":
		return mergeHeadsShape(dimensions)
	case "shape.concat":
		return concatShape(inputShapes, configInt(node.Config, "dim", 0))
	case "shape.transpose":
		return transposeShape(
			dimensions,
			configInt(node.Config, "dim0", 0),
			configInt(node.Config, "dim1", 0),
		)
	case "shape.reshape":
		if shape := configIntSlice(node.Config, "shape"); len(shape) > 0 {
			return tensor.NewShape(shape)
		}

		if shape := configIntSlice(node.Config, "target_shape"); len(shape) > 0 {
			return tensor.NewShape(shape)
		}
	case "shape.upsample_nearest2d":
		return upsampleNearest2DShape(dimensions, node.Config)
	case "convolution.conv2d":
		return conv2DShape(dimensions, node.Config)
	case "convolution.conv_transpose2d":
		return convTranspose2DShape(dimensions, node.Config)
	}

	return opShape, nil
}

func appendShapeDim(dimensions []int, dimension int) (tensor.Shape, error) {
	if dimension <= 0 {
		return tensor.Shape{}, fmt.Errorf("manifest: inferred dimension must be positive, got %d", dimension)
	}

	out := append(append([]int(nil), dimensions...), dimension)

	return tensor.NewShape(out)
}

func concatShape(inputShapes []tensor.Shape, dim int) (tensor.Shape, error) {
	if len(inputShapes) == 0 {
		return tensor.Shape{}, fmt.Errorf("manifest: shape.concat requires at least one input")
	}

	dimensions := inputShapes[0].Dims()

	if dim < 0 || dim >= len(dimensions) {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: shape.concat dim %d out of range rank %d",
			dim,
			len(dimensions),
		)
	}

	out := append([]int(nil), dimensions...)

	for inputIndex, inputShape := range inputShapes[1:] {
		inputDimensions := inputShape.Dims()

		if len(inputDimensions) != len(dimensions) {
			return tensor.Shape{}, fmt.Errorf(
				"manifest: shape.concat input %d rank %d does not match rank %d",
				inputIndex+1,
				len(inputDimensions),
				len(dimensions),
			)
		}

		for dimensionIndex, dimension := range inputDimensions {
			if dimensionIndex == dim {
				out[dim] += dimension

				continue
			}

			if dimension != dimensions[dimensionIndex] {
				return tensor.Shape{}, fmt.Errorf(
					"manifest: shape.concat input %d dimension %d is %d, expected %d",
					inputIndex+1,
					dimensionIndex,
					dimension,
					dimensions[dimensionIndex],
				)
			}
		}
	}

	return tensor.NewShape(out)
}

func transposeShape(dimensions []int, dim0 int, dim1 int) (tensor.Shape, error) {
	if dim0 < 0 || dim0 >= len(dimensions) || dim1 < 0 || dim1 >= len(dimensions) {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: shape.transpose dims %d,%d out of range rank %d",
			dim0,
			dim1,
			len(dimensions),
		)
	}

	out := append([]int(nil), dimensions...)
	out[dim0], out[dim1] = out[dim1], out[dim0]

	return tensor.NewShape(out)
}

func replaceLastShapeDim(dimensions []int, dimension int) (tensor.Shape, error) {
	if dimension <= 0 {
		return tensor.Shape{}, fmt.Errorf("manifest: inferred dimension must be positive, got %d", dimension)
	}

	if len(dimensions) == 0 {
		return tensor.NewShape([]int{dimension})
	}

	out := append([]int(nil), dimensions...)
	out[len(out)-1] = dimension

	return tensor.NewShape(out)
}

func viewAsHeadsShape(dimensions []int, numHeads int) (tensor.Shape, error) {
	if numHeads <= 0 {
		return tensor.Shape{}, fmt.Errorf("manifest: num_heads must be positive, got %d", numHeads)
	}

	if len(dimensions) != 3 {
		return tensor.NewShape(dimensions)
	}

	if dimensions[2]%numHeads != 0 {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: hidden dimension %d is not divisible by num_heads %d",
			dimensions[2], numHeads,
		)
	}

	return tensor.NewShape([]int{dimensions[0], numHeads, dimensions[1], dimensions[2] / numHeads})
}

func mergeHeadsShape(dimensions []int) (tensor.Shape, error) {
	if len(dimensions) != 4 {
		return tensor.NewShape(dimensions)
	}

	return tensor.NewShape([]int{dimensions[0], dimensions[2], dimensions[1] * dimensions[3]})
}

func upsampleNearest2DShape(dimensions []int, config map[string]any) (tensor.Shape, error) {
	if len(dimensions) != 4 {
		return tensor.NewShape(dimensions)
	}

	scaleH := configIntAny(config, 0, "scale_h", "scale_factor")
	scaleW := configIntAny(config, 0, "scale_w", "scale_factor")
	outH := configIntAny(config, 0, "out_h", "height")
	outW := configIntAny(config, 0, "out_w", "width")

	if scaleH == 0 && outH > 0 {
		if outH%dimensions[2] != 0 {
			return tensor.Shape{}, fmt.Errorf(
				"manifest: shape.upsample_nearest2d out_h %d is not divisible by input height %d",
				outH,
				dimensions[2],
			)
		}

		scaleH = outH / dimensions[2]
	}

	if scaleW == 0 && outW > 0 {
		if outW%dimensions[3] != 0 {
			return tensor.Shape{}, fmt.Errorf(
				"manifest: shape.upsample_nearest2d out_w %d is not divisible by input width %d",
				outW,
				dimensions[3],
			)
		}

		scaleW = outW / dimensions[3]
	}

	if scaleH <= 0 || scaleW <= 0 {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: shape.upsample_nearest2d scale_h and scale_w must be positive",
		)
	}

	return tensor.NewShape([]int{
		dimensions[0],
		dimensions[1],
		dimensions[2] * scaleH,
		dimensions[3] * scaleW,
	})
}

func conv2DShape(dimensions []int, config map[string]any) (tensor.Shape, error) {
	if len(dimensions) != 4 {
		return tensor.NewShape(dimensions)
	}

	outChannels := configIntAny(config, 0, "out_channels", "out_c")
	kernelH := configIntAny(config, 0, "kernel_h", "k_h")
	kernelW := configIntAny(config, 0, "kernel_w", "k_w")
	strideH := configIntAny(config, 1, "stride_h", "s_h")
	strideW := configIntAny(config, 1, "stride_w", "s_w")
	padH := configIntAny(config, 0, "pad_h", "p_h")
	padW := configIntAny(config, 0, "pad_w", "p_w")
	dilationH := configIntAny(config, 1, "dil_h", "d_h")
	dilationW := configIntAny(config, 1, "dil_w", "d_w")

	if outChannels <= 0 || kernelH <= 0 || kernelW <= 0 ||
		strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 {
		return tensor.Shape{}, fmt.Errorf("manifest: convolution.conv2d dimensions must be positive")
	}

	heightOut := (dimensions[2]+2*padH-dilationH*(kernelH-1)-1)/strideH + 1
	widthOut := (dimensions[3]+2*padW-dilationW*(kernelW-1)-1)/strideW + 1

	if heightOut <= 0 || widthOut <= 0 {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: convolution.conv2d output shape [%d,%d] must be positive",
			heightOut,
			widthOut,
		)
	}

	return tensor.NewShape([]int{dimensions[0], outChannels, heightOut, widthOut})
}

func convTranspose2DShape(dimensions []int, config map[string]any) (tensor.Shape, error) {
	if len(dimensions) != 4 {
		return tensor.NewShape(dimensions)
	}

	outChannels := configIntAny(config, 0, "out_channels", "out_c")
	kernelH := configIntAny(config, 0, "kernel_h", "k_h")
	kernelW := configIntAny(config, 0, "kernel_w", "k_w")
	strideH := configIntAny(config, 1, "stride_h", "s_h")
	strideW := configIntAny(config, 1, "stride_w", "s_w")
	padH := configIntAny(config, 0, "pad_h", "p_h")
	padW := configIntAny(config, 0, "pad_w", "p_w")
	outPadH := configInt(config, "out_pad_h", 0)
	outPadW := configInt(config, "out_pad_w", 0)
	dilationH := configIntAny(config, 1, "dil_h", "d_h")
	dilationW := configIntAny(config, 1, "dil_w", "d_w")

	if outChannels <= 0 || kernelH <= 0 || kernelW <= 0 ||
		strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: convolution.conv_transpose2d dimensions must be positive",
		)
	}

	heightOut := (dimensions[2]-1)*strideH - 2*padH + dilationH*(kernelH-1) + outPadH + 1
	widthOut := (dimensions[3]-1)*strideW - 2*padW + dilationW*(kernelW-1) + outPadW + 1

	if heightOut <= 0 || widthOut <= 0 {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: convolution.conv_transpose2d output shape [%d,%d] must be positive",
			heightOut,
			widthOut,
		)
	}

	return tensor.NewShape([]int{dimensions[0], outChannels, heightOut, widthOut})
}

func lastTokenShape(dimensions []int) (tensor.Shape, error) {
	if len(dimensions) < 2 {
		return tensor.Shape{}, fmt.Errorf(
			"manifest: shape.last_token expects rank >= 2, got %d",
			len(dimensions),
		)
	}

	out := append([]int(nil), dimensions...)
	out[len(out)-2] = 1

	return tensor.NewShape(out)
}

func configInt(config map[string]any, key string, fallback int) int {
	value, ok := config[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return fallback
	}
}

func configIntAny(config map[string]any, fallback int, keys ...string) int {
	for _, key := range keys {
		if value := configInt(config, key, fallback); value != fallback {
			return value
		}
	}

	return fallback
}

func configIntSlice(config map[string]any, key string) []int {
	value, ok := config[key]

	if !ok {
		return nil
	}

	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...)
	case []any:
		out := make([]int, len(typed))

		for index, item := range typed {
			out[index] = configInt(map[string]any{"value": item}, "value", 0)
		}

		return out
	default:
		return nil
	}
}

func attributeFromValue(value any) ir.Attribute {
	switch typed := value.(type) {
	case string:
		return ir.StringAttribute(typed)
	case int:
		return ir.IntAttribute(int64(typed))
	case int64:
		return ir.IntAttribute(typed)
	case float64:
		return ir.FloatAttribute(typed)
	case bool:
		return ir.BoolAttribute(typed)
	default:
		return ir.StringAttribute(fmt.Sprintf("%v", value))
	}
}
