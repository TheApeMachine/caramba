package ir

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
OpType specifies the type of mathematical operation for the node.
*/
type OpType string

const (
	OpInput  OpType = "Input"
	OpMatmul OpType = "Matmul"
	OpAdd    OpType = "Add"
	OpReLU   OpType = "ReLU"
	OpGELU   OpType = "GELU"
	OpFused  OpType = "Fused"
)

/*
Node represents an operation in the intermediate representation graph.
It abstracts the hardware-specific implementation so operations can be routed generically.
*/
type Node struct {
	ctx      context.Context
	cancel   context.CancelFunc
	err      error
	id       string
	opType   OpType
	shape    tensor.Shape
	inPlace  bool
	inputs   []*Node
	metadata map[string]any
}

/*
NewNode instantiates a new Node.
It serves as a single mathematical step in a larger compute graph.
*/
func NewNode(ctx context.Context, id string, opType OpType, shape tensor.Shape) *Node {
	ctx, cancel := context.WithCancel(ctx)

	return &Node{
		ctx:      ctx,
		cancel:   cancel,
		id:       id,
		opType:   opType,
		shape:    shape,
		inputs:   make([]*Node, 0),
		metadata: make(map[string]any),
	}
}

/*
ID returns the node's unique identifier.
*/
func (node *Node) ID() string {
	return node.id
}

/*
OpType returns the node's operation type.
*/
func (node *Node) OpType() OpType {
	return node.opType
}

/*
Shape returns the node's output shape.
*/
func (node *Node) Shape() tensor.Shape {
	return node.shape
}

/*
Inputs returns the nodes that this node depends on.
*/
func (node *Node) Inputs() []*Node {
	return node.inputs
}

/*
AddInput adds a dependency to this node.
*/
func (node *Node) AddInput(input *Node) {
	node.inputs = append(node.inputs, input)
}

/*
Metadata returns additional configuration for the node.
*/
func (node *Node) Metadata() map[string]any {
	return node.metadata
}

/*
SetMetadata adds configuration to the node.
*/
func (node *Node) SetMetadata(key string, value any) {
	node.metadata[key] = value
}

/*
InPlace returns whether the node should mutate its input buffer.
*/
func (node *Node) InPlace() bool {
	return node.inPlace
}

/*
SetInPlace configures whether the node can safely mutate its input buffer.
*/
func (node *Node) SetInPlace(val bool) {
	node.inPlace = val
}
