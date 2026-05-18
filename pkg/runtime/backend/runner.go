package backend

import (
	"context"
	"fmt"
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
	"github.com/theapemachine/caramba/pkg/manifest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
GraphRunner is the bridge between the runtime executor and the
compute backend. It implements op.GraphRunner.Call by resolving a
program.GraphModule to a manifest.Graph, lowering it to backend IR,
binding the runtime-supplied inputs onto IR input nodes, and
executing the resulting graph through the supplied compute.Backend.

The runner caches resolved manifest.Graph instances by manifest path
so repeated calls reuse a single compile. The IR graph itself is
rebuilt per call because input shapes can vary between iterations.

WeightBinder and PreExecute hooks let callers add policy on top of
the bridge without forking the runner. Runtime model adapters use
them to bind parameters and attach graph-declared state inputs such
as KV-cache and RoPE position metadata.
*/
type GraphRunner struct {
	compute          *compute.Backend
	manifestCompiler *manifest.Compiler
	defaultManifest  string
	weightBinder     WeightBinder
	preExecute       PreExecuteHook
	defaultPrecision dtype.DType

	mu    sync.Mutex
	cache map[string]*manifest.Graph
}

/*
WeightBinder runs once per IR graph (per call) before the graph is
executed. It receives the freshly lowered IR so the binder can locate
parameter nodes by ID and attach the appropriate tensor values via
node metadata.
*/
type WeightBinder func(irGraph *ir.Graph, module program.GraphModule) error

/*
PreExecuteHook runs after input binding and weight binding, just
before the backend executes the IR. It is the slot for per-call
metadata injection driven by graph.call inputs.
*/
type PreExecuteHook func(irGraph *ir.Graph, inputs map[string]any) error

/*
Options configures a GraphRunner. ComputeBackend is required;
ProjectRoot anchors manifest paths; DefaultManifest is consulted
when a GraphModule does not carry its own Manifest path; Preloaded
maps a logical manifest key (typically the path) to an already
compiled manifest.Graph, which lets callers feed graphs produced
from embedded assets or from custom resolvers without forcing the
runner to encode that policy. DefaultPrecision, when set, forces
every IR node's ValueType.Precision after lowering — the standard
way to let a manifest written at float64 lower onto a Metal or
CUDA backend that executes at float32.
*/
type Options struct {
	ComputeBackend   *compute.Backend
	ProjectRoot      string
	DefaultManifest  string
	WeightBinder     WeightBinder
	PreExecute       PreExecuteHook
	Preloaded        map[string]*manifest.Graph
	DefaultPrecision dtype.DType
}

func New(options Options) (*GraphRunner, error) {
	if options.ComputeBackend == nil {
		return nil, fmt.Errorf("runtime/backend: compute backend is required")
	}

	root := options.ProjectRoot

	if root == "" {
		root = "."
	}

	cache := map[string]*manifest.Graph{}

	for key, graph := range options.Preloaded {
		cache[key] = graph
	}

	return &GraphRunner{
		compute:          options.ComputeBackend,
		manifestCompiler: manifest.NewCompiler(root),
		defaultManifest:  options.DefaultManifest,
		weightBinder:     options.WeightBinder,
		preExecute:       options.PreExecute,
		defaultPrecision: options.DefaultPrecision,
		cache:            cache,
	}, nil
}

/*
Preload caches a manifest.Graph by key so subsequent Call invocations
that reference the same key skip disk lookup and compilation.
*/
func (graphRunner *GraphRunner) Preload(key string, graph *manifest.Graph) {
	graphRunner.mu.Lock()
	defer graphRunner.mu.Unlock()

	graphRunner.cache[key] = graph
}

/*
Call satisfies op.GraphRunner. The runtime executor calls this from
graph.call ops.
*/
func (graphRunner *GraphRunner) Call(
	callContext context.Context,
	module program.GraphModule,
	inputs map[string]any,
) (map[string]any, error) {
	graph, err := graphRunner.resolveGraph(module)

	if err != nil {
		return nil, fmt.Errorf("graph %q: %w", module.ID, err)
	}

	defaultShape, err := deriveDefaultShape(module, inputs)

	if err != nil {
		return nil, fmt.Errorf("graph %q: %w", module.ID, err)
	}

	inputShapes, err := deriveInputShapes(module, inputs)

	if err != nil {
		return nil, fmt.Errorf("graph %q: %w", module.ID, err)
	}

	irGraph, err := manifest.LowerGraphToIRWithInputShapes(graph, defaultShape, inputShapes)

	if err != nil {
		return nil, fmt.Errorf("graph %q: lowering: %w", module.ID, err)
	}

	index, err := irGraph.Index()

	if err != nil {
		return nil, fmt.Errorf("graph %q: indexing: %w", module.ID, err)
	}

	if err := bindAllInputs(index, inputs); err != nil {
		return nil, fmt.Errorf("graph %q: binding inputs: %w", module.ID, err)
	}

	if graphRunner.defaultPrecision != dtype.Invalid {
		applyDefaultPrecision(irGraph, graphRunner.defaultPrecision)
	}

	if graphRunner.weightBinder != nil {
		if err := graphRunner.weightBinder(irGraph, module); err != nil {
			return nil, fmt.Errorf("graph %q: binding weights: %w", module.ID, err)
		}
	}

	if graphRunner.preExecute != nil {
		if err := graphRunner.preExecute(irGraph, inputs); err != nil {
			return nil, fmt.Errorf("graph %q: pre-execute hook: %w", module.ID, err)
		}
	}

	targets, outputNames, err := resolveTargets(module, irGraph)

	if err != nil {
		return nil, fmt.Errorf("graph %q: resolving outputs: %w", module.ID, err)
	}

	if err := validateBoundInputsUsed(index, inputs, targets); err != nil {
		return nil, fmt.Errorf("graph %q: %w", module.ID, err)
	}

	executionOutputs, err := graphRunner.compute.Execute(callContext, irGraph, targets)

	if err != nil {
		return nil, fmt.Errorf("graph %q: executing: %w", module.ID, err)
	}

	return collectOutputs(targets, outputNames, executionOutputs)
}

func (graphRunner *GraphRunner) resolveGraph(
	module program.GraphModule,
) (*manifest.Graph, error) {
	graphRunner.mu.Lock()

	if module.Topology != "" {
		if cached, ok := graphRunner.cache[module.Topology]; ok {
			graphRunner.mu.Unlock()

			return cached, nil
		}
	}

	path := module.Manifest

	if path == "" {
		path = graphRunner.defaultManifest
	}

	if path == "" {
		graphRunner.mu.Unlock()

		if module.Topology != "" {
			return nil, fmt.Errorf(
				"topology %q not preloaded and no manifest path declared",
				module.Topology,
			)
		}

		return nil, fmt.Errorf("no manifest path declared and no default configured")
	}

	if cached, ok := graphRunner.cache[path]; ok {
		graphRunner.mu.Unlock()

		return cached, nil
	}

	graphRunner.mu.Unlock()

	compiled, err := graphRunner.manifestCompiler.Compile(path)

	if err != nil {
		return nil, err
	}

	graphRunner.mu.Lock()
	graphRunner.cache[path] = compiled
	graphRunner.mu.Unlock()

	return compiled, nil
}

func collectOutputs(
	targets []*ir.Node, outputNames []string, executionOutputs map[string]tensor.Tensor,
) (map[string]any, error) {
	results := make(map[string]any, len(targets))

	defer func() {
		for _, outputTensor := range executionOutputs {
			if outputTensor != nil {
				_ = outputTensor.Close()
			}
		}
	}()

	for index, target := range targets {
		outputTensor, ok := executionOutputs[target.ID()]

		if !ok || outputTensor == nil {
			return nil, fmt.Errorf("target %q produced no output", target.ID())
		}

		sourceDType, bytes, err := outputTensor.RawBytes()

		if err != nil {
			return nil, fmt.Errorf("target %q clone: %w", target.ID(), err)
		}

		values, err := dtypeconvert.BytesToFloat64(sourceDType, bytes)

		if err != nil {
			return nil, fmt.Errorf("target %q clone: %w", target.ID(), err)
		}

		results[outputNames[index]] = values
	}

	return results, nil
}
