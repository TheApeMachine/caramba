package runtime

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	runtimebackend "github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	"github.com/theapemachine/caramba/pkg/runtime/op"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/provenance"
	"github.com/theapemachine/caramba/pkg/runtime/sampler"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
RuntimeGenerator executes a manifest-driven token program.
*/
type RuntimeGenerator struct {
	program       *program.Program
	tokenizer     tokenizer.Tokenizer
	tokenizerName string
	graphRunner   op.GraphRunner
	samplerRunner op.SamplerRunner
	telemetry     telemetry.Recorder
}

type RuntimeGeneratorOptions struct {
	Program          *program.Program
	Tokenizer        tokenizer.Tokenizer
	TokenizerAssetID string
	GraphRunner      op.GraphRunner
	SamplerRunner    op.SamplerRunner
	SamplerSeed      uint64
	ProjectRoot      string
	ManifestPath     string
	Telemetry        telemetry.Recorder
}

func NewRuntimeGenerator(options RuntimeGeneratorOptions) (*RuntimeGenerator, error) {
	if options.Tokenizer == nil {
		return nil, fmt.Errorf("runtime/model: tokenizer is required")
	}

	if options.GraphRunner == nil {
		return nil, fmt.Errorf("runtime/model: graph runner is required")
	}

	runtimeProgram, err := resolveProgram(options)

	if err != nil {
		return nil, err
	}

	samplerRunner := options.SamplerRunner

	if samplerRunner == nil {
		samplerRunner = sampler.New(options.SamplerSeed)
	}

	tokenizerAsset := options.TokenizerAssetID

	if tokenizerAsset == "" {
		tokenizerAsset = "tokenizer"
	}

	return &RuntimeGenerator{
		program:       runtimeProgram,
		tokenizer:     options.Tokenizer,
		tokenizerName: tokenizerAsset,
		graphRunner:   options.GraphRunner,
		samplerRunner: samplerRunner,
		telemetry:     options.Telemetry,
	}, nil
}

func (generator *RuntimeGenerator) Generate(
	ctx context.Context,
	prompt string,
	emit func(string) error,
) error {
	if emit == nil {
		return fmt.Errorf("runtime/model: emit callback is required")
	}

	stdin := strings.NewReader(prompt + "\n")
	writer := &emitWriter{emit: emit}

	exec, err := generator.newExecutor(stdin, writer)

	if err != nil {
		return err
	}

	if err := exec.Run(ctx); err != nil {
		return err
	}

	return writer.err
}

func (generator *RuntimeGenerator) RunSession(
	ctx context.Context,
	input io.Reader,
	output io.Writer,
) error {
	exec, err := generator.newExecutor(input, output)

	if err != nil {
		return err
	}

	return exec.Run(ctx)
}

func (generator *RuntimeGenerator) newExecutor(
	stdin io.Reader,
	stdout io.Writer,
) (*executor.Executor, error) {
	return executor.New(executor.Options{
		Program:       generator.program,
		Tokenizers:    map[string]tokenizer.Tokenizer{generator.tokenizerName: generator.tokenizer},
		GraphRunner:   generator.graphRunner,
		SamplerRunner: generator.samplerRunner,
		Telemetry:     generator.telemetry,
		Stdin:         stdin,
		Stdout:        stdout,
	})
}

func resolveProgram(options RuntimeGeneratorOptions) (*program.Program, error) {
	if options.Program != nil {
		return options.Program, nil
	}

	root := options.ProjectRoot

	if root == "" {
		root = "."
	}

	runtimeCompiler := compiler.New(root)

	if options.ManifestPath != "" {
		return runtimeCompiler.Compile(options.ManifestPath)
	}

	data, err := asset.ReadFile(defaultChatRuntimeAsset)

	if err != nil {
		return nil, fmt.Errorf("runtime/model: load default runtime manifest: %w", err)
	}

	return runtimeCompiler.CompileBytes(data)
}

type RuntimeModelGenerator struct {
	inner     *RuntimeGenerator
	backend   *compute.Backend
	model     string
	telemetry telemetry.Recorder
	ledger    *provenance.Ledger
}

func NewRuntimeModelGenerator(
	ctx context.Context,
	config ModelConfig,
) (*RuntimeModelGenerator, error) {
	resolved, err := resolveManifestModelConfig(config)

	if err != nil {
		return nil, err
	}

	if err := validateModelRuntime(resolved); err != nil {
		return nil, err
	}

	computeBackend, err := resolved.ComputeBackend()

	if err != nil {
		return nil, err
	}

	topologyGraph, modelManifestPath, err := compileModelManifest(resolved.Manifest)

	if err != nil {
		return nil, err
	}

	tokenizerSource := tokenizerSource(resolved)

	if tokenizerSource.Source == "" {
		return nil, fmt.Errorf("runtime/model: tokenizer source is required")
	}

	tokenizerArtifact, err := tokenizer.Load(ctx, tokenizerSource)

	if err != nil {
		return nil, err
	}

	weightStore, err := modelweights.Resolve(ctx, modelWeightSource(resolved))

	if err != nil {
		return nil, err
	}

	runtimeProgram, runtimeManifestPath, err := loadRuntimeProgram(
		resolved.RuntimeManifest,
		defaultChatRuntimeAsset,
		"runtime/model",
	)

	if err != nil {
		return nil, err
	}

	graphRunner, err := runtimebackend.New(runtimebackend.Options{
		ComputeBackend:   computeBackend,
		WeightBinder:     NewWeightBinder(weightStore),
		PreExecute:       runtimebackend.NewStateBindingHook(),
		Preloaded:        topologyPreload(runtimeProgram, topologyGraph),
		DefaultPrecision: backendComputePrecision(computeBackend.Location()),
	})

	if err != nil {
		return nil, err
	}

	recorder := telemetry.NewInMemory()
	ledger := newModelLedger(resolved, modelManifestPath, runtimeManifestPath, weightStore, tokenizerArtifact)

	inner, err := NewRuntimeGenerator(RuntimeGeneratorOptions{
		Program:          runtimeProgram,
		Tokenizer:        tokenizerArtifact.Tokenizer,
		GraphRunner:      graphRunner,
		SamplerRunner:    sampler.New(uint64(resolved.Seed)),
		TokenizerAssetID: "tokenizer",
		Telemetry:        recorder,
	})

	if err != nil {
		return nil, err
	}

	return &RuntimeModelGenerator{
		inner:     inner,
		backend:   computeBackend,
		model:     resolved.Model,
		telemetry: recorder,
		ledger:    ledger,
	}, nil
}

func (generator *RuntimeModelGenerator) Generate(
	ctx context.Context,
	prompt string,
	emit func(string) error,
) error {
	if generator == nil || generator.inner == nil {
		return fmt.Errorf("runtime/model: generator is not initialized")
	}

	return generator.inner.Generate(ctx, prompt, emit)
}

func (generator *RuntimeModelGenerator) RunSession(
	ctx context.Context,
	input io.Reader,
	output io.Writer,
) error {
	if generator == nil || generator.inner == nil {
		return fmt.Errorf("runtime/model: generator is not initialized")
	}

	return generator.inner.RunSession(ctx, input, output)
}

func (generator *RuntimeModelGenerator) BackendName() string {
	if generator == nil || generator.backend == nil {
		return ""
	}

	return string(generator.backend.Location())
}

func (generator *RuntimeModelGenerator) ModelName() string {
	if generator == nil {
		return ""
	}

	return generator.model
}

func (generator *RuntimeModelGenerator) Telemetry() telemetry.Recorder {
	if generator == nil {
		return nil
	}

	return generator.telemetry
}

func (generator *RuntimeModelGenerator) Ledger() *provenance.Ledger {
	if generator == nil {
		return nil
	}

	return generator.ledger
}

func (generator *RuntimeModelGenerator) WriteLedger(path string) error {
	if generator == nil || generator.ledger == nil {
		return fmt.Errorf("runtime/model: ledger is not initialized")
	}

	foldTelemetry(generator.ledger, generator.telemetry)

	return generator.ledger.WriteFile(path)
}
