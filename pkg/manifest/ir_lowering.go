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
	nodes := make(map[string]*ir.Node, len(graph.nodes))

	for _, manifestNode := range graph.Nodes() {
		opID := manifestNode.OpID
		if opID == "" {
			opID = manifestNode.ID
		}

		node := ir.NewNode(manifestNode.ID, ir.OpType(opID), defaultShape)
		node.SetOperationID(ir.OpID(opID))

		for key, value := range manifestNode.Config {
			node.SetMetadata(key, value)
			node.SetAttribute(key, attributeFromValue(value))
		}

		if precision, ok := manifestNode.Config["precision"].(string); ok {
			node.SetValueType(ir.ValueType{
				Shape:     defaultShape,
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
		irGraph.AddNode(node)
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
