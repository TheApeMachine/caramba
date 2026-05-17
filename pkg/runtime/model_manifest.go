package runtime

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/manifest"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

const (
	modelManifestPrefix     = "runtime/model"
	modelDefaultRuntime     = "model"
	modelDefaultBackend     = "auto"
	modelDefaultRepoType    = "model"
	defaultChatRuntimeAsset = "runtime/chat.yml"
)

func resolveManifestModelConfig(config ModelConfig) (ModelConfig, error) {
	document, manifestPath, err := parseModelManifestDocument(config.Manifest)

	if err != nil {
		return config, err
	}

	config.Manifest = manifestPath
	runtimeBlock, ok, err := modelManifestRuntime(document)

	if err != nil || !ok {
		return config, err
	}

	applyModelDefaults(&config)

	if err := applyModelRuntime(&config, runtimeBlock); err != nil {
		return config, err
	}

	return config, nil
}

func modelManifestRuntime(
	document map[string]any,
) (map[string]any, bool, error) {
	return optionalMap(document, modelManifestPrefix, "system", "runtime")
}

func applyModelDefaults(config *ModelConfig) {
	config.Runtime = modelDefaultRuntime
	config.Backend = modelDefaultBackend
	config.RepoType = modelDefaultRepoType
	config.ModelRepoType = modelDefaultRepoType
	config.TokenizerRepoType = modelDefaultRepoType
}

func validateModelRuntime(config ModelConfig) error {
	runtimeName := strings.ToLower(strings.TrimSpace(config.Runtime))

	if runtimeName == "" || runtimeName == modelDefaultRuntime {
		return nil
	}

	return fmt.Errorf("runtime/model: unsupported manifest runtime %q", config.Runtime)
}

func applyModelRuntime(config *ModelConfig, runtimeBlock map[string]any) error {
	if runtimeName, ok, err := optionalString(runtimeBlock, modelManifestPrefix, "type"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Runtime = runtimeName
	}

	if backendName, ok, err := optionalString(runtimeBlock, modelManifestPrefix, "backend"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Backend = backendName
	}

	if err := applyModelSource(config, runtimeBlock); err != nil {
		return err
	}

	if err := applyTokenizerSource(config, runtimeBlock); err != nil {
		return err
	}

	if generationBlock, ok, err := optionalMap(runtimeBlock, modelManifestPrefix, "generation"); ok || err != nil {
		if err != nil {
			return err
		}

		return applyModelGeneration(config, generationBlock)
	}

	return nil
}

func applyModelGeneration(config *ModelConfig, generationBlock map[string]any) error {
	value, ok, err := optionalInt64(generationBlock, modelManifestPrefix, "seed")

	if err != nil {
		return err
	}

	if ok {
		config.Seed = value
	}

	return nil
}

func applyModelSource(config *ModelConfig, runtimeBlock map[string]any) error {
	modelSource, ok, err := optionalSource(runtimeBlock, "model", modelManifestPrefix)

	if err != nil || !ok {
		return err
	}

	config.Model = modelSource.source
	config.ModelFile = modelSource.file
	config.ModelCache = modelSource.cache
	config.ModelRevision = modelSource.revision
	config.ModelRepoType = firstText(modelSource.repoType, modelDefaultRepoType)

	return nil
}

func applyTokenizerSource(config *ModelConfig, runtimeBlock map[string]any) error {
	tokenizerSource, ok, err := optionalSource(runtimeBlock, "tokenizer", modelManifestPrefix)

	if err != nil {
		return err
	}

	if ok {
		config.Tokenizer = tokenizerSource.source
		config.TokenizerFile = tokenizerSource.file
		config.TokenizerCache = tokenizerSource.cache
		config.TokenizerRevision = tokenizerSource.revision
		config.TokenizerRepoType = firstText(tokenizerSource.repoType, modelDefaultRepoType)

		return nil
	}

	config.Tokenizer = config.Model
	config.TokenizerCache = config.ModelCache
	config.TokenizerRevision = config.ModelRevision
	config.TokenizerRepoType = config.ModelRepoType

	return nil
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
		return nil, path, fmt.Errorf("runtime/model: manifest %s: %w", path, err)
	}

	graph, err = manifest.NewCompiler(".").CompileBytes(data)

	if err != nil {
		return nil, path, err
	}

	return graph, path, nil
}

func parseModelManifestDocument(path string) (map[string]any, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultModelManifest
	}

	document, err := parseLocalManifestDocument(path)

	if err == nil {
		return document, path, nil
	}

	if !errors.Is(err, os.ErrNotExist) {
		return nil, path, err
	}

	data, err := asset.ReadFile(path)

	if err != nil {
		return nil, path, fmt.Errorf("runtime/model: manifest %s: %w", path, err)
	}

	document, err = manifest.NewParser(".").ParseBytes(data)

	if err != nil {
		return nil, path, err
	}

	return document, path, nil
}

func parseLocalManifestDocument(path string) (map[string]any, error) {
	info, err := os.Stat(path)

	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return nil, fmt.Errorf("runtime/model: manifest %s is a directory", path)
	}

	if filepath.IsAbs(path) {
		return manifest.NewParser(filepath.Dir(path)).Parse(filepath.Base(path))
	}

	return manifest.NewParser(".").Parse(path)
}

func compileLocalManifest(path string) (*manifest.Graph, error) {
	info, err := os.Stat(path)

	if err != nil {
		return nil, err
	}

	if info.IsDir() {
		return nil, fmt.Errorf("runtime/model: manifest %s is a directory", path)
	}

	if filepath.IsAbs(path) {
		return manifest.NewCompiler(filepath.Dir(path)).Compile(filepath.Base(path))
	}

	return manifest.NewCompiler(".").Compile(path)
}
