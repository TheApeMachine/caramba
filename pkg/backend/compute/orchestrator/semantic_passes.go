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

		if err := validatePrecision(pass.capabilities, node); err != nil {
			return PassResult{}, err
		}

		if err := validateShapeConstraints(pass.capabilities, node); err != nil {
			return PassResult{}, err
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

func validatePrecision(capabilities Capabilities, node *ir.Node) error {
	if !node.IsPure() {
		return nil
	}

	// OpInput is a data source, not a compute step. Storage precision
	// (DType) drives upload conversions; compute precision is only
	// meaningful for actual math kernels. Validating Input here would
	// reject any manifest that authors values as float64 even though
	// the backend's UploadFloat64 path converts to its native dtype
	// faithfully before any kernel runs.
	if node.OpType() == ir.OpInput {
		return nil
	}

	required := node.ValueType().Precision

	if required == "" {
		required = node.ValueType().DType
	}

	if required == "" {
		required = "float64"
	}

	actual := capabilities.Precision(node.OpType())

	if required == actual {
		return nil
	}

	return fmt.Errorf(
		"lowering: backend %s executes %q at %s precision, but node %q requires %s precision",
		capabilities.Location(), node.OpType(), actual, node.ID(), required,
	)
}

/*
ShapeConstraint identifiers used by backends through the
shapeConstraintProvider interface. Centralizing these strings here
avoids typos between the provider implementations and the dispatch
switch in validateShapeConstraint.
*/
const (
	ShapeConstraintInputsSameShape          = "inputs.same_shape"
	ShapeConstraintMatMulRank2              = "matmul.rank2"
	ShapeConstraintInputRank3               = "input.rank3"
	ShapeConstraintInputRank4               = "input.rank4"
	ShapeConstraintOutputSameElementsInput0 = "output.same_elements_as_input0"
)

type shapeConstraintProvider interface {
	ShapeConstraints(operationID ir.OpType) []string
}

func validateShapeConstraints(capabilities Capabilities, node *ir.Node) error {
	provider, ok := capabilities.(shapeConstraintProvider)

	if !ok {
		return nil
	}

	for _, constraint := range provider.ShapeConstraints(node.OpType()) {
		if err := validateShapeConstraint(capabilities, node, constraint); err != nil {
			return err
		}
	}

	return nil
}

func validateShapeConstraint(capabilities Capabilities, node *ir.Node, constraint string) error {
	switch constraint {
	case ShapeConstraintInputsSameShape:
		return validateSameInputShapes(capabilities, node)
	case ShapeConstraintMatMulRank2:
		return validateRank2Matmul(capabilities, node)
	case ShapeConstraintInputRank3:
		return validateFirstInputRank(capabilities, node, 3)
	case ShapeConstraintInputRank4:
		return validateFirstInputRank(capabilities, node, 4)
	case ShapeConstraintOutputSameElementsInput0:
		return validateSameElementCount(capabilities, node)
	}

	return fmt.Errorf(
		"lowering: backend %s has unknown shape constraint %q for node %q",
		capabilities.Location(), constraint, node.ID(),
	)
}

func validateSameInputShapes(capabilities Capabilities, node *ir.Node) error {
	inputs := node.Inputs()

	if len(inputs) == 0 {
		return fmt.Errorf(
			"lowering: backend %s requires inputs for shape-constrained node %q",
			capabilities.Location(), node.ID(),
		)
	}

	expectedShape := inputs[0].Shape()

	for _, input := range inputs[1:] {
		if input.Shape().Equal(expectedShape) {
			continue
		}

		return fmt.Errorf(
			"lowering: backend %s requires same-shape inputs for node %q",
			capabilities.Location(), node.ID(),
		)
	}

	if node.Shape().Equal(expectedShape) {
		return nil
	}

	return fmt.Errorf(
		"lowering: backend %s requires node %q output shape to match its inputs",
		capabilities.Location(), node.ID(),
	)
}

func validateRank2Matmul(capabilities Capabilities, node *ir.Node) error {
	inputs := node.Inputs()

	if len(inputs) != 2 {
		return fmt.Errorf(
			"lowering: backend %s requires rank-2 matmul node %q to have 2 inputs",
			capabilities.Location(), node.ID(),
		)
	}

	leftShape := inputs[0].Shape().Dims()
	rightShape := inputs[1].Shape().Dims()
	outputShape := node.Shape().Dims()

	if len(leftShape) == 2 && len(rightShape) == 2 && len(outputShape) == 2 &&
		leftShape[1] == rightShape[0] &&
		outputShape[0] == leftShape[0] &&
		outputShape[1] == rightShape[1] {
		return nil
	}

	return fmt.Errorf(
		"lowering: backend %s requires rank-2 matmul shapes for node %q",
		capabilities.Location(), node.ID(),
	)
}

func validateFirstInputRank(capabilities Capabilities, node *ir.Node, rank int) error {
	inputs := node.Inputs()

	if len(inputs) == 0 {
		return fmt.Errorf(
			"lowering: backend %s requires input rank %d for node %q",
			capabilities.Location(), rank, node.ID(),
		)
	}

	if len(inputs[0].Shape().Dims()) == rank {
		return nil
	}

	return fmt.Errorf(
		"lowering: backend %s requires input rank %d for node %q",
		capabilities.Location(), rank, node.ID(),
	)
}

func validateSameElementCount(capabilities Capabilities, node *ir.Node) error {
	inputs := node.Inputs()

	if len(inputs) == 0 {
		return fmt.Errorf(
			"lowering: backend %s requires an input for node %q element-count validation",
			capabilities.Location(), node.ID(),
		)
	}

	if inputs[0].Shape().Len() == node.Shape().Len() {
		return nil
	}

	return fmt.Errorf(
		"lowering: backend %s requires node %q to preserve element count",
		capabilities.Location(), node.ID(),
	)
}
