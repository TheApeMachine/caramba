package orchestrator

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
CSEOptimizer analyzes an intermediate representation graph to find Common Subexpressions.
If multiple nodes perform the exact same mathematical operation on the exact same inputs,
the optimizer folds them into a single evaluation path to save execution time.
*/
type CSEOptimizer struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

/*
NewCSEOptimizer instantiates a new Common Subexpression Elimination optimizer.
*/
func NewCSEOptimizer(ctx context.Context) *CSEOptimizer {
	ctx, cancel := context.WithCancel(ctx)

	return &CSEOptimizer{
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Optimize detects structural equivalence and eliminates redundant calculations.
*/
func (optimizer *CSEOptimizer) Optimize(graph *ir.Graph) *ir.Graph {
	optimizedGraph := ir.NewGraph(optimizer.ctx)

	// signature maps a structural hash to a node ID
	signatures := make(map[string]string)
	replacements := make(map[string]*ir.Node)

	// This relies on nodes being topologically sorted, or at least processing
	// inputs before dependents. ir.Graph currently doesn't enforce topological sort on Nodes()
	// iteration, but for CSE to work correctly, we process iteratively until no changes occur.
	// For simplicity in this architectural pass, we'll do a simple loop.

	for _, node := range graph.Nodes() {
		// Input nodes are exempt from CSE folding directly unless identical shape/metadata?
		// Actually, inputs are typically unique entry points.
		if node.OpType() == ir.OpInput {
			newNode := ir.NewNode(optimizer.ctx, node.ID(), node.OpType(), node.Shape())
			for k, v := range node.Metadata() {
				newNode.SetMetadata(k, v)
			}
			replacements[node.ID()] = newNode
			optimizedGraph.AddNode(newNode)
			continue
		}

		sig := generateSignature(node, replacements)
		if existingID, found := signatures[sig]; found {
			// Redundant calculation found! Point to the existing node.
			replacements[node.ID()] = replacements[existingID]
		} else {
			signatures[sig] = node.ID()

			newNode := ir.NewNode(optimizer.ctx, node.ID(), node.OpType(), node.Shape())
			newNode.SetInPlace(node.InPlace())
			for k, v := range node.Metadata() {
				newNode.SetMetadata(k, v)
			}

			// Add remapped inputs
			for _, in := range node.Inputs() {
				if rep, ok := replacements[in.ID()]; ok {
					newNode.AddInput(rep)
				} else {
					newNode.AddInput(in) // Fallback, though ideally already processed
				}
			}

			replacements[node.ID()] = newNode
			optimizedGraph.AddNode(newNode)
		}
	}

	return optimizedGraph
}

func generateSignature(node *ir.Node, replacements map[string]*ir.Node) string {
	var sb strings.Builder
	sb.WriteString(string(node.OpType()))
	sb.WriteString("|")

	// Include shape in signature
	sb.WriteString(fmt.Sprintf("%v", node.Shape().Dims()))
	sb.WriteString("|")

	for _, in := range node.Inputs() {
		// Use the replacement ID to ensure transitive redundancy is caught
		if rep, ok := replacements[in.ID()]; ok {
			sb.WriteString(rep.ID())
		} else {
			sb.WriteString(in.ID())
		}
		sb.WriteString(",")
	}

	return sb.String()
}
