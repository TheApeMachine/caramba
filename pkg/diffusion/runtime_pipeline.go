package diffusion

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/provenance"
	"github.com/theapemachine/caramba/pkg/runtime/scheduler"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
RuntimeDiffusionPipeline is the manifest-driven diffusion adapter.
It loads the runtime program from runtime/diffusion.yml (or an
override), resolves three model manifests (text_encoder, denoiser,
vae), preloads each topology under the name the runtime program
references, attaches per-graph weight stores via the bridge's
WeightBinder hook, and runs the executor. There is no Go-coded
diffusion control flow in this adapter.
*/
type RuntimeDiffusionPipeline struct {
	program     *program.Program
	tokenizer   tokenizer.Tokenizer
	graphRunner *backend.GraphRunner
	telemetry   telemetry.Recorder
	ledger      *provenance.Ledger
	backend     *compute.Backend
	outputPath  string
}

/*
Result describes the artifact a diffusion run produced.
*/
type Result struct {
	Output string
	Width  int
	Height int
}

const (
	diffusionRuntimeAssetPath = "runtime/diffusion.yml"
	diffusionTokenizerAsset   = "tokenizer"
)

/*
backendComputePrecision returns the compute precision the diffusion
runtime adapter declares on every IR node before execution. GPU
backends (Metal/CUDA/XLA) execute at Float32; CPU at Float64.
UploadFloat64 paths convert host values into the device's storage
dtype before kernels run.
*/
func backendComputePrecision(location tensor.Location) tensor.DType {
	switch location {
	case tensor.Metal, tensor.CUDA, tensor.XLA:
		return tensor.Float32
	}

	return ""
}

/*
NewRuntimeDiffusionPipeline constructs the runtime-backed diffusion
adapter. config.Manifest names the umbrella diffusion manifest that
declares each component's source; config.Runtime, when set, points at
an alternate runtime manifest.
*/
func NewRuntimeDiffusionPipeline(
	ctx context.Context, config Config,
) (*RuntimeDiffusionPipeline, error) {
	resolved, _, _, err := ResolveManifestConfig(config)

	if err != nil {
		return nil, err
	}

	if err := resolved.ValidateRuntime(); err != nil {
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

	runtimeProgram, runtimeManifestPath, err := loadDiffusionRuntimeProgram(resolved.RuntimeManifest)

	if err != nil {
		return nil, err
	}

	applyRuntimeOverrides(runtimeProgram, resolved)

	// DefaultPrecision lines up the manifest's nodes with the
	// per-backend compute precision: Float32 on GPU backends where
	// UploadFloat64 already converts host values into device storage,
	// empty (Float64) on CPU. See chat.runtime_model_generator for
	// the same contract.
	graphRunner, err := backend.New(backend.Options{
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

/*
Generate runs the diffusion program for prompt and returns the
resulting image's location and dimensions.
*/
func (pipeline *RuntimeDiffusionPipeline) Generate(
	ctx context.Context, prompt string,
) (Result, error) {
	if pipeline == nil || pipeline.program == nil {
		return Result{}, fmt.Errorf("diffusion/runtime: pipeline is not initialized")
	}

	stdin := strings.NewReader(strings.TrimSpace(prompt) + "\n")

	exec, err := executor.New(executor.Options{
		Program:         pipeline.program,
		Tokenizers:      map[string]tokenizer.Tokenizer{diffusionTokenizerAsset: pipeline.tokenizer},
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

/*
Close releases backend resources.
*/
func (pipeline *RuntimeDiffusionPipeline) Close() error {
	if pipeline == nil || pipeline.backend == nil {
		return nil
	}

	return pipeline.backend.Close()
}

/*
Telemetry exposes the in-memory recorder.
*/
func (pipeline *RuntimeDiffusionPipeline) Telemetry() telemetry.Recorder {
	if pipeline == nil {
		return nil
	}

	return pipeline.telemetry
}

/*
Ledger exposes the provenance ledger.
*/
func (pipeline *RuntimeDiffusionPipeline) Ledger() *provenance.Ledger {
	if pipeline == nil {
		return nil
	}

	return pipeline.ledger
}

/*
WriteLedger folds telemetry counters into the ledger and persists it.
*/
func (pipeline *RuntimeDiffusionPipeline) WriteLedger(path string) error {
	if pipeline == nil || pipeline.ledger == nil {
		return fmt.Errorf("diffusion/runtime: ledger is not initialized")
	}

	pipeline.foldTelemetry()

	return pipeline.ledger.WriteFile(path)
}

func (pipeline *RuntimeDiffusionPipeline) foldTelemetry() {
	memory, ok := pipeline.telemetry.(*telemetry.InMemory)

	if !ok {
		return
	}

	for _, name := range memory.CounterNames() {
		pipeline.ledger.RecordEvent("telemetry.counter", map[string]any{
			"name":  name,
			"value": memory.Counter(name),
		})
	}
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
	ctx context.Context, config Config,
) (map[string]*manifest.Graph, map[string]*modelweights.Store, error) {
	specs := []struct {
		ID     string
		Source Source
	}{
		{ID: "text_encoder", Source: config.TextEncoder},
		{ID: "denoiser", Source: config.Transformer},
		{ID: "vae", Source: config.VAE},
	}

	topologies := map[string]*manifest.Graph{}
	stores := map[string]*modelweights.Store{}

	for _, spec := range specs {
		if strings.TrimSpace(spec.Source.Manifest) == "" {
			return nil, nil, fmt.Errorf(
				"diffusion/runtime: %s manifest path is required", spec.ID,
			)
		}

		graph, _, err := CompileManifest(spec.Source.Manifest)

		if err != nil {
			return nil, nil, fmt.Errorf("diffusion/runtime: compile %s manifest: %w", spec.ID, err)
		}

		store, err := modelweights.Resolve(ctx, weightSource(spec.Source))

		if err != nil {
			return nil, nil, fmt.Errorf("diffusion/runtime: resolve %s weights: %w", spec.ID, err)
		}

		topologies[spec.ID] = graph
		stores[spec.ID] = store
	}

	return topologies, stores, nil
}

func loadDiffusionTokenizer(
	ctx context.Context, config Config,
) (*tokenizer.Artifact, error) {
	source := tokenizerSource(config.Tokenizer)

	if source.Source == "" {
		return nil, fmt.Errorf("diffusion/runtime: tokenizer source is required")
	}

	return tokenizer.Load(ctx, source)
}

func loadDiffusionRuntimeProgram(path string) (*program.Program, string, error) {
	path = strings.TrimSpace(path)

	if path != "" {
		runtimeCompiler := compiler.New(filepath.Dir(path))
		runtimeProgram, err := runtimeCompiler.Compile(filepath.Base(path))

		if err == nil {
			return runtimeProgram, path, nil
		}

		if !os.IsNotExist(err) {
			return nil, path, err
		}
	}

	assetPath := path

	if assetPath == "" {
		assetPath = diffusionRuntimeAssetPath
	}

	data, err := asset.ReadFile(assetPath)

	if err != nil {
		return nil, assetPath, fmt.Errorf("diffusion/runtime: load runtime manifest %s: %w", assetPath, err)
	}

	runtimeProgram, err := compiler.New(".").CompileBytes(data)

	if err != nil {
		return nil, assetPath, err
	}

	return runtimeProgram, assetPath, nil
}

/*
applyRuntimeOverrides folds Config.Generation values into the
compiled runtime program. Output path and seed are the two things
the CLI honours per-invocation; everything else lives in the YAML.
*/
func applyRuntimeOverrides(runtimeProgram *program.Program, config Config) {
	if writeStep := runtimeProgram.FindStep("write_image"); writeStep != nil {
		if path := strings.TrimSpace(config.Generation.Output); path != "" {
			writeStep.Config["path"] = path
		}
	}
}

func newDiffusionLedger(
	config Config,
	runtimeManifestPath string,
	stores map[string]*modelweights.Store,
	tokenizerArtifact *tokenizer.Artifact,
) *provenance.Ledger {
	ledger := provenance.New(map[string]any{
		"program":          "diffusion",
		"backend":          config.Backend,
		"manifest":         config.Manifest,
		"runtime_manifest": runtimeManifestPath,
		"output":           config.Generation.Output,
	})

	ledger.RecordAsset("manifest", config.Manifest, "")
	ledger.RecordAsset("runtime_manifest", runtimeManifestPath, "")
	ledger.RecordAsset("tokenizer", tokenizerArtifact.Path, "")

	for id, store := range stores {
		ledger.RecordAsset(id, fmt.Sprintf("safetensors:%d-tensors", len(store.Names())), "")
	}

	ledger.RecordSeed("main", config.Generation.Seed)

	return ledger
}

