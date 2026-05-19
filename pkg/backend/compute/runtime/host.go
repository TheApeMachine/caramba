package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

/*
Host executes IR graphs against tensor.Host resident storage.
*/
type Host struct {
	memory tensor.Backend
}

func NewHost(memory tensor.Backend) *Host {
	return &Host{memory: memory}
}

func (host *Host) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (map[string]tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if graph == nil {
		return nil, fmt.Errorf("host executor: graph is required")
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("host executor: no execution targets provided")
	}

	if err := validateTargets(graph, targets); err != nil {
		return nil, err
	}

	layers, err := graph.TopologyLayers()

	if err != nil {
		return nil, err
	}

	values := make(map[string]tensor.Tensor, len(graph.Nodes()))

	for _, layer := range layers {
		for _, node := range layer {
			if err := ctx.Err(); err != nil {
				closeValues(values)

				return nil, err
			}

			value, err := host.evaluateNode(node, values)

			if err != nil {
				closeValues(values)

				return nil, err
			}

			if value != nil {
				values[node.ID()] = value
			}
		}
	}

	keep := targetSet(targets)
	outputs := make(map[string]tensor.Tensor, len(targets))

	for _, target := range targets {
		value, ok := values[target.ID()]

		if !ok || value == nil {
			closeValues(values)

			return nil, fmt.Errorf("host executor: target %q produced no output", target.ID())
		}

		outputs[target.ID()] = value
	}

	for nodeID, value := range values {
		if keep[nodeID] {
			continue
		}

		if err := value.Close(); err != nil {
			return nil, err
		}
	}

	return outputs, nil
}

func (host *Host) evaluateNode(
	node *ir.Node,
	values map[string]tensor.Tensor,
) (tensor.Tensor, error) {
	switch node.OpType() {
	case ir.OpInput:
		return host.materializeInput(node)
	default:
		return nil, fmt.Errorf("host executor: op %q is not implemented yet", node.OpType())
	}
}

func (host *Host) materializeInput(node *ir.Node) (tensor.Tensor, error) {
	rawValues, ok := node.Metadata()["values"]

	if !ok {
		return nil, fmt.Errorf("host executor: input node %s has no values metadata", node.ID())
	}

	values, ok := rawValues.([]float64)

	if !ok {
		return nil, fmt.Errorf("host executor: input node %s values metadata must be []float64", node.ID())
	}

	bytes := dtypeconvert.Float64ToBytes(values)

	return host.memory.Upload(node.Shape(), dtype.Float64, bytes)
}

func (host *Host) Location() tensor.Location {
	return tensor.Host
}

func (host *Host) Close() error {
	if host == nil || host.memory == nil {
		return nil
	}

	return host.memory.Close()
}

var _ Executor = (*Host)(nil)

func validateTargets(graph *ir.Graph, targets []*ir.Node) error {
	index, err := graph.Index()

	if err != nil {
		return err
	}

	for _, target := range targets {
		if target == nil {
			return fmt.Errorf("host executor: nil execution target")
		}

		if index.Node(target.ID()) == nil {
			return fmt.Errorf("host executor: target node %q is not registered in graph", target.ID())
		}
	}

	return nil
}

func targetSet(targets []*ir.Node) map[string]bool {
	set := make(map[string]bool, len(targets))

	for _, target := range targets {
		set[target.ID()] = true
	}

	return set
}

func closeValues(values map[string]tensor.Tensor) {
	for _, value := range values {
		if value != nil {
			_ = value.Close()
		}
	}
}
