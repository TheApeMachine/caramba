package runtime

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	runtimebackend "github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/provenance"
	"github.com/theapemachine/caramba/pkg/runtime/telemetry"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

func backendComputePrecision(location tensor.Location) dtype.DType {
	switch location {
	case tensor.Metal, tensor.CUDA, tensor.XLA:
		return dtype.Float32
	}

	return dtype.Invalid
}

func NewWeightBinder(store *modelweights.Store) runtimebackend.WeightBinder {
	if store == nil {
		return nil
	}

	return func(irGraph *ir.Graph, module program.GraphModule) error {
		return modelweights.BindIR(irGraph, store)
	}
}

func loadRuntimeProgram(path string, assetPath string, prefix string) (*program.Program, string, error) {
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

	resolvedAssetPath := path

	if resolvedAssetPath == "" {
		resolvedAssetPath = assetPath
	}

	data, err := asset.ReadFile(resolvedAssetPath)

	if err != nil {
		return nil, resolvedAssetPath, fmt.Errorf(
			"%s: load runtime manifest %s: %w",
			prefix,
			resolvedAssetPath,
			err,
		)
	}

	runtimeProgram, err := compiler.New(".").CompileBytes(data)

	if err != nil {
		return nil, resolvedAssetPath, err
	}

	return runtimeProgram, resolvedAssetPath, nil
}

func topologyPreload(
	runtimeProgram *program.Program,
	topology *manifest.Graph,
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

func newModelLedger(
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

func foldTelemetry(ledger *provenance.Ledger, recorder telemetry.Recorder) {
	memory, ok := recorder.(*telemetry.InMemory)

	if !ok {
		return
	}

	for _, name := range memory.CounterNames() {
		ledger.RecordEvent("telemetry.counter", map[string]any{
			"name":  name,
			"value": memory.Counter(name),
		})
	}
}

type emitWriter struct {
	emit func(string) error
	err  error
}

func (writer *emitWriter) Write(payload []byte) (int, error) {
	if writer.err != nil {
		return 0, writer.err
	}

	if err := writer.emit(string(payload)); err != nil {
		writer.err = err

		return 0, err
	}

	return len(payload), nil
}
