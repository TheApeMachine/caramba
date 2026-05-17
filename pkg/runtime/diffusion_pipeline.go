package runtime

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	runtimebackend "github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/provenance"
	"github.com/theapemachine/caramba/pkg/runtime/scheduler"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
RuntimeDiffusionPipeline executes an image runtime program.
*/
type RuntimeDiffusionPipeline struct {
	program     *program.Program
	tokenizer   tokenizer.Tokenizer
	graphRunner *runtimebackend.GraphRunner
	telemetry   telemetry.Recorder
	ledger      *provenance.Ledger
	backend     *compute.Backend
	outputPath  string
}

func NewRuntimeDiffusionPipeline(
	ctx context.Context,
	config DiffusionConfig,
) (*RuntimeDiffusionPipeline, error) {
	resolved, _, _, err := ResolveDiffusionConfig(config)

	if err != nil {
		return nil, err
	}

	if err := validateDiffusionRuntime(resolved); err != nil {
		return nil, err
	}

	computeBackend, err := resolved.ComputeBackend()

	if err != nil {
		return nil, err
	}

	topologies, weightStores, err := loadDiffusionAssets(ctx, resolved)

	if err != nil {
		return nil, err
	}

	tokenizerArtifact, err := loadDiffusionTokenizer(ctx, resolved)

	if err != nil {
		return nil, err
	}

	runtimeProgram, runtimeManifestPath, err := loadRuntimeProgram(
		resolved.RuntimeManifest,
		diffusionRuntimeAsset,
		"runtime/diffusion",
	)

	if err != nil {
		return nil, err
	}

	applyRuntimeOverrides(runtimeProgram, resolved)

	graphRunner, err := runtimebackend.New(runtimebackend.Options{
		ComputeBackend:   computeBackend,
		WeightBinder:     newWeightDispatch(weightStores),
		Preloaded:        topologies,
		DefaultPrecision: backendComputePrecision(computeBackend.Location()),
	})

	if err != nil {
		return nil, err
	}

	recorder := telemetry.NewInMemory()
	ledger := newDiffusionLedger(resolved, runtimeManifestPath, weightStores, tokenizerArtifact)

	return &RuntimeDiffusionPipeline{
		program:     runtimeProgram,
		tokenizer:   tokenizerArtifact.Tokenizer,
		graphRunner: graphRunner,
		telemetry:   recorder,
		ledger:      ledger,
		backend:     computeBackend,
		outputPath:  resolved.Generation.Output,
	}, nil
}

func (pipeline *RuntimeDiffusionPipeline) Generate(
	ctx context.Context,
	prompt string,
) (Result, error) {
	if pipeline == nil || pipeline.program == nil {
		return Result{}, fmt.Errorf("runtime/diffusion: pipeline is not initialized")
	}

	stdin := strings.NewReader(strings.TrimSpace(prompt) + "\n")

	exec, err := executor.New(executor.Options{
		Program:         pipeline.program,
		Tokenizers:      map[string]tokenizer.Tokenizer{"tokenizer": pipeline.tokenizer},
		GraphRunner:     pipeline.graphRunner,
		SchedulerRunner: scheduler.NewFlowMatchEuler(),
		Telemetry:       pipeline.telemetry,
		Stdin:           stdin,
		Stdout:          io.Discard,
	})

	if err != nil {
		return Result{}, err
	}

	if err := exec.Run(ctx); err != nil {
		return Result{}, err
	}

	width, height := pipeline.dimensions()

	return Result{
		Output: pipeline.outputPath,
		Width:  width,
		Height: height,
	}, nil
}

func (pipeline *RuntimeDiffusionPipeline) Close() error {
	if pipeline == nil || pipeline.backend == nil {
		return nil
	}

	return pipeline.backend.Close()
}

func (pipeline *RuntimeDiffusionPipeline) Telemetry() telemetry.Recorder {
	if pipeline == nil {
		return nil
	}

	return pipeline.telemetry
}

func (pipeline *RuntimeDiffusionPipeline) Ledger() *provenance.Ledger {
	if pipeline == nil {
		return nil
	}

	return pipeline.ledger
}

func (pipeline *RuntimeDiffusionPipeline) WriteLedger(path string) error {
	if pipeline == nil || pipeline.ledger == nil {
		return fmt.Errorf("runtime/diffusion: ledger is not initialized")
	}

	foldTelemetry(pipeline.ledger, pipeline.telemetry)

	return pipeline.ledger.WriteFile(path)
}

func (pipeline *RuntimeDiffusionPipeline) dimensions() (int, int) {
	writeStep := pipeline.program.FindStep("write_image")

	if writeStep == nil {
		return 0, 0
	}

	width, _ := writeStep.Config["width"].(int)
	height, _ := writeStep.Config["height"].(int)

	return width, height
}

func loadDiffusionAssets(
	ctx context.Context,
	config DiffusionConfig,
) (map[string]*manifest.Graph, map[string]*modelweights.Store, error) {
	denoiserSource := config.Transformer

	if strings.TrimSpace(denoiserSource.Manifest) == "" {
		denoiserSource.Manifest = config.Manifest
	}

	specs := []struct {
		ID     string
		Source Source
	}{
		{ID: "text_encoder", Source: config.TextEncoder},
		{ID: "denoiser", Source: denoiserSource},
		{ID: "vae", Source: config.VAE},
	}

	topologies := map[string]*manifest.Graph{}
	stores := map[string]*modelweights.Store{}

	for _, spec := range specs {
		if strings.TrimSpace(spec.Source.Manifest) == "" {
			return nil, nil, fmt.Errorf(
				"runtime/diffusion: %s manifest path is required",
				spec.ID,
			)
		}

		graph, _, err := CompileDiffusionManifest(spec.Source.Manifest)

		if err != nil {
			return nil, nil, fmt.Errorf("runtime/diffusion: compile %s manifest: %w", spec.ID, err)
		}

		store, err := modelweights.Resolve(ctx, weightSource(spec.Source))

		if err != nil {
			return nil, nil, fmt.Errorf("runtime/diffusion: resolve %s weights: %w", spec.ID, err)
		}

		topologies[spec.ID] = graph
		stores[spec.ID] = store
	}

	return topologies, stores, nil
}
