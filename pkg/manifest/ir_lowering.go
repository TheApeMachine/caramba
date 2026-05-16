package manifest

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func LowerGraphToIR(graph *Graph, defaultShape tensor.Shape) (*ir.Graph, error) {
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

		node := ir.NewNode(input, ir.OpInput, defaultShape)
		node.SetOperationID(ir.OpID("data.input"))
		node.SetMetadata("binding", input)
		node.SetAttribute("out.0", ir.StringAttribute(input))

		nodes[input] = node
		bindingShapes[input] = defaultShape
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
		outputShape, err := outputShapeForNode(manifestNode, opShape)

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

func outputShapeForNode(node *Node, opShape tensor.Shape) (tensor.Shape, error) {
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
	case "shape.reshape":
		if shape := configIntSlice(node.Config, "shape"); len(shape) > 0 {
			return tensor.NewShape(shape)
		}

		if shape := configIntSlice(node.Config, "target_shape"); len(shape) > 0 {
			return tensor.NewShape(shape)
		}
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
