package orchestrator

import (
	"context"
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

func (optimizer *CSEOptimizer) Name() string {
	return "semantic-cse"
}

func (optimizer *CSEOptimizer) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := ctx.Err(); err != nil {
		return PassResult{}, err
	}

	graph, targets, err := optimizer.OptimizeWithTargets(input.Graph, input.Targets)

	if err != nil {
		return PassResult{}, err
	}

	input.Diagnostics.Add(optimizer.Name(), DiagnosticInfo, "deduplicated semantic expressions")

	return PassResult{
		Graph:       graph,
		Targets:     targets,
		TargetMap:   targetMap(targets),
		Diagnostics: input.Diagnostics,
		Changed:     true,
	}, nil
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
			if node.OpType() == ir.OpInput || !node.IsPure() {
				newNode := ir.NewNode(node.ID(), node.OpType(), node.Shape())
				newNode.SetInPlace(node.InPlace())
				newNode.SetOperationID(node.OperationID())
				newNode.SetValueType(node.ValueType())
				newNode.SetEffect(node.Effect())
				newNode.SetAlias(node.Alias())
				for k, v := range node.Metadata() {
					newNode.SetMetadata(k, v)
				}
				for k, v := range node.Attributes() {
					newNode.SetAttribute(k, v)
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
				newNode.SetOperationID(node.OperationID())
				newNode.SetValueType(node.ValueType())
				newNode.SetEffect(node.Effect())
				newNode.SetAlias(node.Alias())
				for k, v := range node.Metadata() {
					newNode.SetMetadata(k, v)
				}
				for k, v := range node.Attributes() {
					newNode.SetAttribute(k, v)
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
	sb.WriteString(string(node.OperationID()))
	sb.WriteString("|")

	valueType := node.ValueType()
	sb.WriteString(fmt.Sprintf("%v|%s|%s|%s|", node.Shape().Dims(), valueType.DType, valueType.Layout, valueType.MemoryClass))

	if node.InPlace() {
		sb.WriteString("inplace=true|")
	} else {
		sb.WriteString("inplace=false|")
	}

	sb.WriteString(node.CanonicalAttributes())
	sb.WriteString("|")

	inputIDs := canonicalInputIDs(node, replacements)

	for _, inputID := range inputIDs {
		sb.WriteString(inputID)
		sb.WriteString(",")
	}

	return sb.String()
}

func canonicalInputIDs(node *ir.Node, replacements map[string]*ir.Node) []string {
	inputs := node.Inputs()
	inputIDs := make([]string, 0, len(inputs))

	for _, in := range node.Inputs() {
		if rep, ok := replacements[in.ID()]; ok {
			inputIDs = append(inputIDs, rep.ID())
		} else {
			inputIDs = append(inputIDs, in.ID())
		}
	}

	if node.OpType() == ir.OpAdd || node.OpType() == ir.OpMul {
		sort.Strings(inputIDs)
	}

	return inputIDs
}
