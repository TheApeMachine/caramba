package orchestrator

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

type DiagnosticLevel string

const (
	DiagnosticInfo    DiagnosticLevel = "info"
	DiagnosticWarning DiagnosticLevel = "warning"
	DiagnosticError   DiagnosticLevel = "error"
)

type Diagnostic struct {
	Pass    string
	Level   DiagnosticLevel
	Message string
}

type Diagnostics struct {
	messages []Diagnostic
}

func (diagnostics *Diagnostics) Add(pass string, level DiagnosticLevel, message string) {
	diagnostics.messages = append(diagnostics.messages, Diagnostic{
		Pass:    pass,
		Level:   level,
		Message: message,
	})
}

func (diagnostics *Diagnostics) Messages() []Diagnostic {
	messages := make([]Diagnostic, len(diagnostics.messages))
	copy(messages, diagnostics.messages)

	return messages
}

type TargetMap map[string]*ir.Node

type PassInput struct {
	Graph       *ir.Graph
	Targets     []*ir.Node
	TargetMap   TargetMap
	Diagnostics *Diagnostics
}

type PassResult struct {
	Graph       *ir.Graph
	Targets     []*ir.Node
	TargetMap   TargetMap
	Diagnostics *Diagnostics
	Changed     bool
}

type Pass interface {
	Name() string
	Run(ctx context.Context, input PassInput) (PassResult, error)
}

type Pipeline struct {
	passes []Pass
}

func NewPipeline(passes ...Pass) *Pipeline {
	return &Pipeline{passes: passes}
}

func (pipeline *Pipeline) Run(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (PassResult, error) {
	if graph == nil {
		return PassResult{}, fmt.Errorf("pipeline: graph is required")
	}

	targetMap := make(TargetMap, len(targets))
	for _, target := range targets {
		if target != nil {
			targetMap[target.ID()] = target
		}
	}

	result := PassResult{
		Graph:       graph,
		Targets:     targets,
		TargetMap:   targetMap,
		Diagnostics: &Diagnostics{},
	}

	for _, pass := range pipeline.passes {
		if err := ctx.Err(); err != nil {
			return PassResult{}, err
		}

		next, err := pass.Run(ctx, PassInput{
			Graph:       result.Graph,
			Targets:     result.Targets,
			TargetMap:   result.TargetMap,
			Diagnostics: result.Diagnostics,
		})

		if err != nil {
			return PassResult{}, fmt.Errorf("%s: %w", pass.Name(), err)
		}

		result = next
	}

	return result, nil
}

type VerifierPass struct{}

func NewVerifierPass() *VerifierPass {
	return &VerifierPass{}
}

func (pass *VerifierPass) Name() string {
	return "verifier"
}

func (pass *VerifierPass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	if err := ctx.Err(); err != nil {
		return PassResult{}, err
	}

	if err := input.Graph.Verify(); err != nil {
		return PassResult{}, err
	}

	if err := validateTargets(input.Graph, input.Targets); err != nil {
		return PassResult{}, err
	}

	input.Diagnostics.Add(pass.Name(), DiagnosticInfo, "verified graph")

	return PassResult{
		Graph:       input.Graph,
		Targets:     input.Targets,
		TargetMap:   input.TargetMap,
		Diagnostics: input.Diagnostics,
	}, nil
}
