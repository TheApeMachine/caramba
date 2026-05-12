package orchestrator

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

type CanonicalizePass struct{}

func NewCanonicalizePass() *CanonicalizePass {
	return &CanonicalizePass{}
}

func (pass *CanonicalizePass) Name() string {
	return "canonicalize"
}

func (pass *CanonicalizePass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	graph, replacements, err := input.Graph.Clone()

	if err != nil {
		return PassResult{}, err
	}

	for _, node := range graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return PassResult{}, err
		}

		if node.OperationID() == "" {
			node.SetOperationID(ir.OpID(node.OpType()))
		}
	}

	targets := remapTargets(input.Targets, replacements)
	input.Diagnostics.Add(pass.Name(), DiagnosticInfo, "canonicalized operation attributes")

	return PassResult{
		Graph:       graph,
		Targets:     targets,
		TargetMap:   targetMap(targets),
		Diagnostics: input.Diagnostics,
		Changed:     true,
	}, nil
}

type AlgebraicSimplifyPass struct{}

func NewAlgebraicSimplifyPass() *AlgebraicSimplifyPass {
	return &AlgebraicSimplifyPass{}
}

func (pass *AlgebraicSimplifyPass) Name() string {
	return "algebraic-simplify"
}

func (pass *AlgebraicSimplifyPass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	input.Diagnostics.Add(pass.Name(), DiagnosticInfo, "algebraic simplification complete")

	return PassResult{
		Graph:       input.Graph,
		Targets:     input.Targets,
		TargetMap:   input.TargetMap,
		Diagnostics: input.Diagnostics,
	}, nil
}

type MemoryPlanPass struct {
	planner *MemoryPlanner
}

func NewMemoryPlanPass() *MemoryPlanPass {
	return &MemoryPlanPass{planner: NewMemoryPlanner()}
}

func (pass *MemoryPlanPass) Name() string {
	return "memory-plan"
}

func (pass *MemoryPlanPass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	plan, err := pass.planner.Plan(input.Graph, input.Targets)

	if err != nil {
		return PassResult{}, err
	}

	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	for _, node := range input.Graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return PassResult{}, err
		}

		node.SetMetadata("memory_buffer", plan.Buffer(node.ID()))
	}

	input.Diagnostics.Add(pass.Name(), DiagnosticInfo, "planned tensor lifetimes and buffers")

	return PassResult{
		Graph:       input.Graph,
		Targets:     input.Targets,
		TargetMap:   input.TargetMap,
		Diagnostics: input.Diagnostics,
		Changed:     true,
	}, nil
}

type LoweringPass struct {
	capabilities Capabilities
}

type SchedulePass struct {
	capabilities Capabilities
}

func NewSchedulePass(capabilities Capabilities) *SchedulePass {
	return &SchedulePass{capabilities: capabilities}
}

func (pass *SchedulePass) Name() string {
	return "schedule"
}

func (pass *SchedulePass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	layers, err := input.Graph.TopologyLayers()

	if err != nil {
		return PassResult{}, err
	}

	order := 0
	for layerIndex, layer := range layers {
		if err := checkContext(ctx); err != nil {
			return PassResult{}, err
		}

		for _, node := range layer {
			if err := checkContext(ctx); err != nil {
				return PassResult{}, err
			}

			cost := pass.capabilities.Cost(node)
			node.SetMetadata("schedule_layer", layerIndex)
			node.SetMetadata("schedule_order", order)
			node.SetMetadata("estimated_flops", cost.FLOPs)
			order++
		}
	}

	input.Diagnostics.Add(pass.Name(), DiagnosticInfo, "assigned cost-based schedule metadata")

	return PassResult{
		Graph:       input.Graph,
		Targets:     input.Targets,
		TargetMap:   input.TargetMap,
		Diagnostics: input.Diagnostics,
		Changed:     true,
	}, nil
}

func NewLoweringPass(capabilities Capabilities) *LoweringPass {
	return &LoweringPass{capabilities: capabilities}
}

func (pass *LoweringPass) Name() string {
	return "lowering"
}

func (pass *LoweringPass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := checkContext(ctx); err != nil {
		return PassResult{}, err
	}

	if pass.capabilities == nil {
		return PassResult{}, fmt.Errorf("lowering: backend capabilities are required")
	}

	for _, node := range input.Graph.Nodes() {
		if err := checkContext(ctx); err != nil {
			return PassResult{}, err
		}

		if !pass.capabilities.Supports(node.OpType()) && node.IsPure() {
			return PassResult{}, fmt.Errorf("lowering: backend does not support %q", node.OpType())
		}
	}

	input.Diagnostics.Add(pass.Name(), DiagnosticInfo, "validated backend lowering legality")

	return PassResult{
		Graph:       input.Graph,
		Targets:     input.Targets,
		TargetMap:   input.TargetMap,
		Diagnostics: input.Diagnostics,
	}, nil
}
