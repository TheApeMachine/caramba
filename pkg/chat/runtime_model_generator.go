package chat

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/provenance"
	"github.com/theapemachine/caramba/pkg/runtime/sampler"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
RuntimeModelGenerator is a thin adapter on top of the runtime
executor. It compiles a runtime manifest (the chat program) and a
model topology manifest (the transformer graph), preloads the
topology into the backend GraphRunner under the name the runtime
program references, and hands the result to a RuntimeGenerator.

This package contains no chat program construction in Go — every
sampling, looping, and streaming decision is declared in
pkg/asset/template/runtime/chat.yml (or a user-supplied override).
*/
type RuntimeModelGenerator struct {
	inner     *RuntimeGenerator
	backend   *compute.Backend
	model     string
	telemetry telemetry.Recorder
	ledger    *provenance.Ledger
}

/*
NewRuntimeModelGenerator wires the runtime program from a chat
runtime manifest to a backend-bound GraphRunner. config.Manifest
points at the MODEL manifest (transformer topology + weights);
config.Runtime, when non-empty, selects a runtime manifest path
override — otherwise the canonical embedded runtime/chat.yml is
used.
*/
func NewRuntimeModelGenerator(
	ctx context.Context, config ModelConfig,
) (*RuntimeModelGenerator, error) {
	resolved, err := resolveManifestModelConfig(config)

	if err != nil {
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
		return nil, fmt.Errorf("chat.runtime: tokenizer source is required")
	}

	tokenizerArtifact, err := tokenizer.Load(ctx, tokenizerSource)

	if err != nil {
		return nil, err
	}

	weightStore, err := modelweights.Resolve(ctx, modelWeightSource(resolved))

	if err != nil {
		return nil, err
	}

	runtimeProgram, runtimeManifestPath, err := loadChatRuntimeProgram(runtimeManifestPath(resolved))

	if err != nil {
		return nil, err
	}

	preload := topologyPreload(runtimeProgram, topologyGraph)

	// DefaultPrecision is the compute-precision the bridge stamps on
	// every IR node before execution. The Metal backend's kernels
	// run at Float32 and the UploadFloat64 path converts host values
	// to that storage faithfully; CPU runs at Float64. Setting this
	// per-backend lines up the manifest's nodes with the kernels.
	graphRunner, err := backend.New(backend.Options{
		ComputeBackend:   computeBackend,
		WeightBinder:     NewWeightBinder(weightStore),
		PreExecute:       NewPreExecuteHook(),
		Preloaded:        preload,
		DefaultPrecision: backendComputePrecision(computeBackend.Location()),
	})

	if err != nil {
		return nil, err
	}

	recorder := telemetry.NewInMemory()
	ledger := newRunLedger(resolved, modelManifestPath, runtimeManifestPath, weightStore, tokenizerArtifact)

	inner, err := NewRuntimeGenerator(RuntimeGeneratorOptions{
		Program:          runtimeProgram,
		Tokenizer:        tokenizerArtifact.Tokenizer,
		GraphRunner:      graphRunner,
		SamplerRunner:    sampler.New(uint64(resolved.Seed)),
		TokenizerAssetID: chatTokenizerAssetID,
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

/*
Generate satisfies the Generator interface. The runtime program
already declares how to format and tokenize the prompt; the prompt
arrives as the stdin line the io.read_line op consumes.
*/
func (generator *RuntimeModelGenerator) Generate(
	ctx context.Context, prompt string, emit func(string) error,
) error {
	if generator == nil || generator.inner == nil {
		return fmt.Errorf("chat.runtime: generator is not initialized")
	}

	return generator.inner.Generate(ctx, prompt, emit)
}

/*
BackendName reports the concrete compute location used by this
generator. Mirrors ModelGenerator.BackendName so cmd/chat keeps
working through the same introspection interface.
*/
func (generator *RuntimeModelGenerator) BackendName() string {
	if generator == nil || generator.backend == nil {
		return ""
	}

	return string(generator.backend.Location())
}

/*
ModelName reports the serialized model source resolved from the
manifest, again mirroring ModelGenerator.
*/
func (generator *RuntimeModelGenerator) ModelName() string {
	if generator == nil {
		return ""
	}

	return generator.model
}

/*
Telemetry exposes the in-memory telemetry recorder so callers can
read counter values, histogram samples, and traced tensors after a
generation completes.
*/
func (generator *RuntimeModelGenerator) Telemetry() telemetry.Recorder {
	if generator == nil {
		return nil
	}

	return generator.telemetry
}

/*
Ledger exposes the run's provenance ledger.
*/
func (generator *RuntimeModelGenerator) Ledger() *provenance.Ledger {
	if generator == nil {
		return nil
	}

	return generator.ledger
}

/*
WriteLedger folds telemetry counters into the ledger and writes the
serialized form to path.
*/
func (generator *RuntimeModelGenerator) WriteLedger(path string) error {
	if generator == nil || generator.ledger == nil {
		return fmt.Errorf("chat.runtime: ledger is not initialized")
	}

	generator.foldTelemetry()

	return generator.ledger.WriteFile(path)
}

func (generator *RuntimeModelGenerator) foldTelemetry() {
	memory, ok := generator.telemetry.(*telemetry.InMemory)

	if !ok {
		return
	}

	for _, name := range memory.CounterNames() {
		generator.ledger.RecordEvent("telemetry.counter", map[string]any{
			"name":  name,
			"value": memory.Counter(name),
		})
	}
}

func newRunLedger(
	config ModelConfig,
	modelManifestPath string,
	runtimeManifestPath string,
	weightStore *modelweights.Store,
	tokenizerArtifact *tokenizer.Artifact,
) *provenance.Ledger {
	ledger := provenance.New(map[string]any{
		"program":          "chat",
		"backend":          config.Backend,
		"model":            config.Model,
		"model_manifest":   modelManifestPath,
		"runtime_manifest": runtimeManifestPath,
	})

	ledger.RecordAsset("model_manifest", modelManifestPath, "")
	ledger.RecordAsset("runtime_manifest", runtimeManifestPath, "")
	ledger.RecordAsset("tokenizer", tokenizerArtifact.Path, "")
	ledger.RecordAsset("model", weightSourceKey(weightStore), "")
	ledger.RecordSeed("main", config.Seed)

	return ledger
}

func weightSourceKey(store *modelweights.Store) string {
	if store == nil {
		return ""
	}

	names := store.Names()

	if len(names) == 0 {
		return ""
	}

	return fmt.Sprintf("safetensors:%d-tensors", len(names))
}

/*
backendComputePrecision returns the compute precision the chat
runtime adapter declares on every IR node before execution. The
Metal/CUDA/XLA backends execute at Float32; CPU executes at
Float64. UploadFloat64 paths on every backend convert host values
into the device's storage dtype before kernels run, so this only
selects the compute-precision contract — not the storage dtype.
*/
func backendComputePrecision(location tensor.Location) tensor.DType {
	switch location {
	case tensor.Metal, tensor.CUDA, tensor.XLA:
		return tensor.Float32
	}

	return ""
}

const (
	chatTokenizerAssetID   = "tokenizer"
	chatRuntimeAssetPath   = "runtime/chat.yml"
	chatDefaultTopologyKey = "model"
)

/*
runtimeManifestPath picks the chat runtime manifest path. The
ModelConfig.Runtime field doubles as the override: a value ending in
".yml" or "/" is treated as a path, anything else falls through to
the embedded default. This is the same convention compileModelManifest
uses for the model manifest.
*/
func runtimeManifestPath(config ModelConfig) string {
	if override := config.RuntimeManifest; override != "" {
		return override
	}

	return ""
}

/*
loadChatRuntimeProgram compiles the runtime program from path when
supplied, or from the embedded runtime/chat.yml otherwise.
*/
func loadChatRuntimeProgram(path string) (*program.Program, string, error) {
	if path != "" {
		runtimeCompiler := compiler.New(filepath.Dir(path))
		runtimeProgram, err := runtimeCompiler.Compile(filepath.Base(path))

		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, path, err
		}

		if err == nil {
			return runtimeProgram, path, nil
		}
	}

	assetPath := path

	if assetPath == "" {
		assetPath = chatRuntimeAssetPath
	}

	data, err := asset.ReadFile(assetPath)

	if err != nil {
		return nil, assetPath, fmt.Errorf("chat.runtime: load runtime manifest %s: %w", assetPath, err)
	}

	runtimeProgram, err := compiler.New(".").CompileBytes(data)

	if err != nil {
		return nil, assetPath, err
	}

	return runtimeProgram, assetPath, nil
}

/*
topologyPreload binds every graph module in the runtime program that
declares a topology reference to the compiled model topology. In the
canonical chat runtime there is exactly one — `forward: { topology:
model }` — but the same shape supports multi-graph programs.
*/
func topologyPreload(
	runtimeProgram *program.Program, topology *manifest.Graph,
) map[string]*manifest.Graph {
	out := map[string]*manifest.Graph{}

	for _, module := range runtimeProgram.Graphs {
		if module.Topology == "" {
			continue
		}

		out[module.Topology] = topology
	}

	return out
}
