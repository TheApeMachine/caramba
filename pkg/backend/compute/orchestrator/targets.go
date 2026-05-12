package orchestrator

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

func remapTargets(targets []*ir.Node, replacements map[string]*ir.Node) []*ir.Node {
	if len(targets) == 0 {
		return nil
	}

	remapped := make([]*ir.Node, len(targets))

	for index, target := range targets {
		if target == nil {
			continue
		}

		replacement, ok := replacements[target.ID()]

		if ok {
			remapped[index] = replacement
			continue
		}

		remapped[index] = target
	}

	return remapped
}

func targetMap(targets []*ir.Node) TargetMap {
	mapping := make(TargetMap, len(targets))

	for _, target := range targets {
		if target != nil {
			mapping[target.ID()] = target
		}
	}

	return mapping
}

func validateTargets(graph *ir.Graph, targets []*ir.Node) error {
	if len(targets) == 0 {
		return fmt.Errorf("scheduler: no execution targets")
	}

	nodes := graph.Nodes()
	nodeByID := make(map[string]*ir.Node, len(nodes))

	for _, node := range nodes {
		nodeByID[node.ID()] = node
	}

	for _, target := range targets {
		if target == nil {
			return fmt.Errorf("scheduler: nil execution target")
		}

		if _, ok := nodeByID[target.ID()]; !ok {
			return fmt.Errorf("scheduler: target %q was removed by optimization", target.ID())
		}
	}

	return nil
}
