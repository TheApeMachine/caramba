package chat

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
DefaultModelManifest is the asset path consulted when ModelConfig
does not name an explicit model manifest.
*/
const DefaultModelManifest = "model/llm/gpt2.yml"

/*
ModelConfig is the minimal set of fields the chat runtime adapter
reads. Everything else (sampling, prompt template, stop policy) is
declared in the runtime manifest itself, not in Go config.
*/
type ModelConfig struct {
	Runtime           string
	RuntimeManifest   string
	Backend           string
	Model             string
	ModelFile         string
	Tokenizer         string
	TokenizerFile     string
	Manifest          string
	Cache             string
	Revision          string
	RepoType          string
	ModelCache        string
	ModelRevision     string
	ModelRepoType     string
	TokenizerCache    string
	TokenizerRevision string
	TokenizerRepoType string
	Seed              int64
}

/*
ComputeBackend resolves the compute backend the runtime adapter will
execute graphs on. An empty Backend keeps callers on CPU until the
manifest sets it.
*/
func (config ModelConfig) ComputeBackend() (*compute.Backend, error) {
	backendName := strings.ToLower(strings.TrimSpace(config.Backend))

	if backendName == "" {
		backendName = "cpu"
	}

	backendType, err := modelBackendType(backendName)

	if err != nil {
		return nil, err
	}

	return compute.NewBackend(backendType)
}

/*
ValidateRuntime accepts the "model" runtime name (legacy compatibility
for manifests that declare `runtime.type: model`) and the empty case.
Any other type is rejected so manifests that name a research-only
runtime mode never silently fall through.
*/
func (config ModelConfig) ValidateRuntime() error {
	runtimeName := strings.ToLower(strings.TrimSpace(config.Runtime))

	if runtimeName == "" || runtimeName == "model" {
		return nil
	}

	return fmt.Errorf("chat: unsupported manifest runtime %q", config.Runtime)
}

func modelBackendType(backendName string) (compute.BackendType, error) {
	switch backendName {
	case "auto":
		return automaticModelBackendType(), nil
	case "cpu", "host":
		return compute.CPU, nil
	case "metal":
		return compute.METAL, nil
	case "cuda":
		return compute.CUDA, nil
	case "xla":
		return compute.XLA, nil
	}

	return compute.CPU, fmt.Errorf("chat: unsupported backend %q", backendName)
}

func automaticModelBackendType() compute.BackendType {
	switch runtime.GOOS {
	case "darwin":
		return compute.METAL
	case "linux":
		return compute.CUDA
	}

	return compute.CPU
}

func tokenizerSource(config ModelConfig) tokenizer.Source {
	source := strings.TrimSpace(config.Tokenizer)

	if source == "" {
		source = strings.TrimSpace(config.Model)
	}

	if source == "" {
		return tokenizer.Source{}
	}

	return tokenizer.Source{
		Source:   source,
		File:     strings.TrimSpace(config.TokenizerFile),
		Cache:    firstText(config.TokenizerCache, config.Cache),
		Revision: firstText(config.TokenizerRevision, config.Revision),
		RepoType: firstText(config.TokenizerRepoType, config.RepoType),
	}
}

func modelWeightSource(config ModelConfig) modelweights.Source {
	source := strings.TrimSpace(config.Model)

	if source == "" {
		source = strings.TrimSpace(config.Tokenizer)
	}

	return modelweights.Source{
		Source:   source,
		File:     strings.TrimSpace(config.ModelFile),
		Cache:    firstText(config.ModelCache, config.Cache),
		Revision: firstText(config.ModelRevision, config.Revision),
		RepoType: firstText(config.ModelRepoType, config.RepoType),
	}
}

func compileModelManifest(path string) (*manifest.Graph, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultModelManifest
	}

	graph, err := compileLocalManifest(path)

	if err == nil {
		return graph, path, nil
	}

	if !errors.Is(err, os.ErrNotExist) {
		return nil, path, err
	}

	data, err := asset.ReadFile(path)

	if err != nil {
		return nil, path, fmt.Errorf("chat: manifest %s: %w", path, err)
	}

	graph, err = manifest.NewCompiler(".").CompileBytes(data)

	if err != nil {
		return nil, path, err
	}

	return graph, path, nil
}

func compileLocalManifest(path string) (*manifest.Graph, error) {
	info, err := os.Stat(path)

	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return nil, fmt.Errorf("chat: manifest %s is a directory", path)
	}

	if filepath.IsAbs(path) {
		return manifest.NewCompiler(filepath.Dir(path)).Compile(filepath.Base(path))
	}

	return manifest.NewCompiler(".").Compile(path)
}
