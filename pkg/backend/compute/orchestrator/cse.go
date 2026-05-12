package orchestrator

import (
	"fmt"
	"sort"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

/*
CSEOptimizer analyzes an intermediate representation graph to find Common Subexpressions.
If multiple nodes perform the exact same mathematical operation on the exact same inputs,
the optimizer folds them into a single evaluation path to save execution time.
*/
type CSEOptimizer struct {
}

/*
NewCSEOptimizer instantiates a new Common Subexpression Elimination optimizer.
*/
func NewCSEOptimizer() *CSEOptimizer {
	return &CSEOptimizer{}
}

/*
Optimize detects structural equivalence and eliminates redundant calculations.
*/
func (optimizer *CSEOptimizer) Optimize(graph *ir.Graph) (*ir.Graph, error) {
	optimizedGraph, _, err := optimizer.optimize(graph)

	return optimizedGraph, err
}

/*
OptimizeWithTargets returns an optimized graph and remaps the requested targets
to their post-CSE representatives.
*/
func (optimizer *CSEOptimizer) OptimizeWithTargets(
	graph *ir.Graph,
	targets []*ir.Node,
) (*ir.Graph, []*ir.Node, error) {
	optimizedGraph, replacements, err := optimizer.optimize(graph)

	if err != nil {
		return nil, nil, err
	}

	return optimizedGraph, remapTargets(targets, replacements), nil
}

func (optimizer *CSEOptimizer) optimize(graph *ir.Graph) (*ir.Graph, map[string]*ir.Node, error) {
	if graph == nil {
		return nil, nil, fmt.Errorf("cse optimizer: nil graph")
	}

	optimizedGraph := ir.NewGraph()

	// signature maps a structural hash to a node ID
	signatures := make(map[string]string)
	replacements := make(map[string]*ir.Node)

	layers, err := graph.TopologyLayers()
	if err != nil {
		return nil, nil, fmt.Errorf("cse optimizer could not sort graph: %w", err)
	}

	for _, layer := range layers {
		for _, node := range layer {
			if node.OpType() == ir.OpInput {
				newNode := ir.NewNode(node.ID(), node.OpType(), node.Shape())
				newNode.SetInPlace(node.InPlace())
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

				newNode := ir.NewNode(node.ID(), node.OpType(), node.Shape())
				newNode.SetInPlace(node.InPlace())
				for k, v := range node.Metadata() {
					newNode.SetMetadata(k, v)
				}

				// Add remapped inputs
				for _, in := range node.Inputs() {
					if rep, ok := replacements[in.ID()]; ok {
						newNode.AddInput(rep)
					} else {
						newNode.AddInput(in) // Fallback
					}
				}

				replacements[node.ID()] = newNode
				optimizedGraph.AddNode(newNode)
			}
		}
	}

	return optimizedGraph, replacements, nil
}

func generateSignature(node *ir.Node, replacements map[string]*ir.Node) string {
	var sb strings.Builder
	sb.WriteString(string(node.OpType()))
	sb.WriteString("|")

	// Include shape in signature
	sb.WriteString(fmt.Sprintf("%v", node.Shape().Dims()))
	sb.WriteString("|")

	if node.InPlace() {
		sb.WriteString("inplace=true|")
	} else {
		sb.WriteString("inplace=false|")
	}

	// Serialize metadata
	meta := node.Metadata()
	keys := make([]string, 0, len(meta))
	for k := range meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		sb.WriteString(fmt.Sprintf("%s=%v;", k, meta[k]))
	}
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
