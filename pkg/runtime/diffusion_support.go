package runtime

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	runtimebackend "github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/provenance"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

func validateDiffusionRuntime(config DiffusionConfig) error {
	runtimeName := strings.ToLower(strings.TrimSpace(config.Runtime))

	if runtimeName == "diffusion" {
		return nil
	}

	return fmt.Errorf("runtime/diffusion: unsupported manifest runtime %q", config.Runtime)
}

func loadDiffusionTokenizer(
	ctx context.Context,
	config DiffusionConfig,
) (*tokenizer.Artifact, error) {
	source := tokenizerSourceFromRuntime(config.Tokenizer)

	if source.Source == "" {
		return nil, fmt.Errorf("runtime/diffusion: tokenizer source is required")
	}

	return tokenizer.Load(ctx, source)
}

func tokenizerSourceFromRuntime(source Source) tokenizer.Source {
	if strings.TrimSpace(source.Source) == "" {
		return tokenizer.Source{}
	}

	return tokenizer.Source{
		Source:   source.Source,
		File:     source.File,
		Cache:    source.Cache,
		Revision: source.Revision,
		RepoType: source.RepoType,
	}
}

func weightSource(source Source) modelweights.Source {
	return modelweights.Source{
		Source:   source.Source,
		File:     source.File,
		Cache:    source.Cache,
		Revision: source.Revision,
		RepoType: source.RepoType,
	}
}

type weightDispatch struct {
	stores map[string]*modelweights.Store
}

func newWeightDispatch(stores map[string]*modelweights.Store) runtimebackend.WeightBinder {
	dispatch := weightDispatch{stores: stores}

	return dispatch.bind
}

func (dispatch weightDispatch) bind(irGraph *ir.Graph, module program.GraphModule) error {
	store, ok := dispatch.stores[module.ID]

	if !ok || store == nil {
		return fmt.Errorf("runtime/diffusion: no weight store registered for graph %q", module.ID)
	}

	return modelweights.BindIR(irGraph, store)
}

func applyRuntimeOverrides(runtimeProgram *program.Program, config DiffusionConfig) {
	if writeStep := runtimeProgram.FindStep("write_image"); writeStep != nil {
		if path := strings.TrimSpace(config.Generation.Output); path != "" {
			writeStep.Config["path"] = path
		}
	}
}

func newDiffusionLedger(
	config DiffusionConfig,
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
