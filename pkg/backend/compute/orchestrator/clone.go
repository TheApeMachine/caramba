package orchestrator

import "github.com/theapemachine/caramba/pkg/backend/compute/ir"

/*
cloneNodeSemantics copies the complete semantic contract of an IR node without
copying edges. Optimizers should use it whenever a node survives a rewrite.
*/
func cloneNodeSemantics(node *ir.Node) *ir.Node {
	clone := ir.NewNode(node.ID(), node.OpType(), node.Shape())
	clone.SetOperationID(node.OperationID())
	clone.SetValueType(node.ValueType())
	clone.SetEffect(node.Effect())
	clone.SetAlias(node.Alias())
	clone.SetInPlace(node.InPlace())

	for key, value := range node.Metadata() {
		clone.SetMetadata(key, value)
	}

	for key, value := range node.Attributes() {
		clone.SetAttribute(key, value)
	}

	return clone
}
