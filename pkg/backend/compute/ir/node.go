package ir

import (
	"sync"

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
Safe for concurrent access.
*/
type Node struct {
	mu       sync.RWMutex
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
func NewNode(id string, opType OpType, shape tensor.Shape) *Node {
	return &Node{
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
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.id
}

/*
OpType returns the node's operation type.
*/
func (node *Node) OpType() OpType {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.opType
}

/*
Shape returns the node's output shape.
*/
func (node *Node) Shape() tensor.Shape {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.shape
}

/*
Inputs returns the nodes that this node depends on.
Returns a defensive copy of the slice.
*/
func (node *Node) Inputs() []*Node {
	node.mu.RLock()
	defer node.mu.RUnlock()
	out := make([]*Node, len(node.inputs))
	copy(out, node.inputs)
	return out
}

/*
AddInput adds a dependency to this node.
*/
func (node *Node) AddInput(input *Node) {
	if input == nil {
		return
	}
	node.mu.Lock()
	defer node.mu.Unlock()
	node.inputs = append(node.inputs, input)
}

/*
Metadata returns additional configuration for the node.
Returns a defensive copy of the map.
*/
func (node *Node) Metadata() map[string]any {
	node.mu.RLock()
	defer node.mu.RUnlock()
	out := make(map[string]any)
	for k, v := range node.metadata {
		out[k] = v
	}
	return out
}

/*
SetMetadata adds configuration to the node.
*/
func (node *Node) SetMetadata(key string, value any) {
	if key == "" {
		return
	}
	node.mu.Lock()
	defer node.mu.Unlock()
	node.metadata[key] = value
}

/*
InPlace returns whether the node should mutate its input buffer.
*/
func (node *Node) InPlace() bool {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.inPlace
}

/*
SetInPlace configures whether the node can safely mutate its input buffer.
*/
func (node *Node) SetInPlace(val bool) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.inPlace = val
}
