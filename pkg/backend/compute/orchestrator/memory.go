package orchestrator

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

type Lifetime struct {
	FirstUse int
	LastUse  int
}

type MemoryPlan struct {
	lifetimes map[string]Lifetime
	buffers   map[string]string
}

type MemoryPlanner struct{}

func NewMemoryPlanner() *MemoryPlanner {
	return &MemoryPlanner{}
}

func (planner *MemoryPlanner) Plan(graph *ir.Graph, targets []*ir.Node) (*MemoryPlan, error) {
	if graph == nil {
		return nil, fmt.Errorf("memory planner: nil graph")
	}

	if err := graph.Verify(); err != nil {
		return nil, err
	}

	nodes := graph.Nodes()
	lifetimes := make(map[string]Lifetime, len(nodes))
	buffers := make(map[string]string, len(nodes))

	for index, node := range nodes {
		lifetimes[node.ID()] = Lifetime{FirstUse: index, LastUse: index}
	}

	for index, node := range nodes {
		for _, input := range node.Inputs() {
			lifetime := lifetimes[input.ID()]
			if index > lifetime.LastUse {
				lifetime.LastUse = index
			}
			lifetimes[input.ID()] = lifetime
		}
	}

	for _, target := range targets {
		if target == nil {
			continue
		}

		lifetime := lifetimes[target.ID()]
		lifetime.LastUse = len(nodes)
		lifetimes[target.ID()] = lifetime
	}

	freeBuffers := make([]string, 0)
	nextBuffer := 0

	for index, node := range nodes {
		for _, prior := range nodes[:index] {
			lifetime := lifetimes[prior.ID()]
			if lifetime.LastUse == index {
				freeBuffers = append(freeBuffers, buffers[prior.ID()])
			}
		}

		if len(freeBuffers) > 0 && node.IsPure() && node.Alias().Kind != ir.AliasInput {
			buffers[node.ID()] = freeBuffers[len(freeBuffers)-1]
			freeBuffers = freeBuffers[:len(freeBuffers)-1]

			continue
		}

		buffers[node.ID()] = fmt.Sprintf("buffer_%d", nextBuffer)
		nextBuffer++
	}

	return &MemoryPlan{lifetimes: lifetimes, buffers: buffers}, nil
}

func (plan *MemoryPlan) Lifetime(id string) Lifetime {
	if plan == nil {
		return Lifetime{}
	}

	return plan.lifetimes[id]
}

func (plan *MemoryPlan) Buffer(id string) string {
	if plan == nil {
		return ""
	}

	return plan.buffers[id]
}
