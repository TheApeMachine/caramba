package chat

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/manifest"
)

const (
	manifestDefaultRuntime  = "model"
	manifestDefaultBackend  = "auto"
	manifestDefaultRepoType = "model"
)

type manifestSource struct {
	source   string
	file     string
	cache    string
	revision string
	repoType string
}

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

	applyManifestRuntimeDefaults(&config)

	if err := applyManifestRuntime(&config, runtimeBlock); err != nil {
		return config, err
	}

	return config, nil
}

func modelManifestRuntime(
	document map[string]any,
) (map[string]any, bool, error) {
	systemBlock, ok, err := optionalMap(document, "system")

	if err != nil || !ok {
		return nil, false, err
	}

	return optionalMap(systemBlock, "runtime")
}

func applyManifestRuntimeDefaults(config *ModelConfig) {
	config.Runtime = manifestDefaultRuntime
	config.Backend = manifestDefaultBackend
	config.RepoType = manifestDefaultRepoType
	config.ModelRepoType = manifestDefaultRepoType
	config.TokenizerRepoType = manifestDefaultRepoType
}

func applyManifestRuntime(config *ModelConfig, runtimeBlock map[string]any) error {
	if runtimeName, ok, err := optionalString(runtimeBlock, "type"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Runtime = runtimeName
	}

	if backendName, ok, err := optionalString(runtimeBlock, "backend"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Backend = backendName
	}

	if err := applyManifestModelSource(config, runtimeBlock); err != nil {
		return err
	}

	if err := applyManifestTokenizerSource(config, runtimeBlock); err != nil {
		return err
	}

	if generationBlock, ok, err := optionalMap(runtimeBlock, "generation"); ok || err != nil {
		if err != nil {
			return err
		}

		return applyManifestGeneration(config, generationBlock)
	}

	return nil
}

/*
applyManifestGeneration reads only the runtime-relevant generation
field — seed, which seeds the sampler RNG. All other knobs
(temperature, top-k, top-p, max_new_tokens, repetition_penalty,
prompt_template, stop_tokens, stop_special_tokens) are declared in
the runtime manifest's samplers / program blocks now and not
honoured from the model manifest.
*/
func applyManifestGeneration(config *ModelConfig, generationBlock map[string]any) error {
	value, ok, err := optionalInt64(generationBlock, "seed")

	if err != nil {
		return err
	}

	if ok {
		config.Seed = value
	}

	return nil
}

func applyManifestModelSource(
	config *ModelConfig,
	runtimeBlock map[string]any,
) error {
	modelSource, ok, err := optionalSource(runtimeBlock, "model")

	if err != nil || !ok {
		return err
	}

	config.Model = modelSource.source
	config.ModelFile = modelSource.file
	config.ModelCache = modelSource.cache
	config.ModelRevision = modelSource.revision
	config.ModelRepoType = firstText(modelSource.repoType, manifestDefaultRepoType)

	return nil
}

func applyManifestTokenizerSource(
	config *ModelConfig,
	runtimeBlock map[string]any,
) error {
	tokenizerSource, ok, err := optionalSource(runtimeBlock, "tokenizer")

	if err != nil {
		return err
	}

	if ok {
		config.Tokenizer = tokenizerSource.source
		config.TokenizerFile = tokenizerSource.file
		config.TokenizerCache = tokenizerSource.cache
		config.TokenizerRevision = tokenizerSource.revision
		config.TokenizerRepoType = firstText(
			tokenizerSource.repoType,
			manifestDefaultRepoType,
		)

		return nil
	}

	config.Tokenizer = config.Model
	config.TokenizerCache = config.ModelCache
	config.TokenizerRevision = config.ModelRevision
	config.TokenizerRepoType = config.ModelRepoType

	return nil
}


func optionalSource(
	mapping map[string]any,
	key string,
) (manifestSource, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return manifestSource{}, false, nil
	}

	switch typed := value.(type) {
	case string:
		return manifestSource{source: strings.TrimSpace(typed)}, true, nil
	case map[string]any:
		return sourceFromMap(typed)
	default:
		return manifestSource{}, false, fmt.Errorf(
			"chat.model: runtime.%s must be a string or mapping, got %T",
			key,
			value,
		)
	}
}

func sourceFromMap(mapping map[string]any) (manifestSource, bool, error) {
	source, _, err := optionalString(mapping, "source", "repo", "id", "path")

	if err != nil {
		return manifestSource{}, false, err
	}

	file, _, err := optionalString(mapping, "file")

	if err != nil {
		return manifestSource{}, false, err
	}

	cache, _, err := optionalString(mapping, "cache")

	if err != nil {
		return manifestSource{}, false, err
	}

	revision, _, err := optionalString(mapping, "revision")

	if err != nil {
		return manifestSource{}, false, err
	}

	repoType, _, err := optionalString(mapping, "repo_type", "repoType")

	if err != nil {
		return manifestSource{}, false, err
	}

	return manifestSource{
		source:   source,
		file:     file,
		cache:    cache,
		revision: revision,
		repoType: repoType,
	}, true, nil
}

func parseModelManifestDocument(
	path string,
) (map[string]any, string, error) {
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
		return nil, path, fmt.Errorf("chat.model: manifest %s: %w", path, err)
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
		return nil, fmt.Errorf("chat.model: manifest %s is a directory", path)
	}

	if filepath.IsAbs(path) {
		return manifest.NewParser(filepath.Dir(path)).Parse(filepath.Base(path))
	}

	return manifest.NewParser(".").Parse(path)
}

func optionalMap(
	mapping map[string]any,
	key string,
) (map[string]any, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return nil, false, nil
	}

	child, ok := value.(map[string]any)

	if !ok {
		return nil, false, fmt.Errorf(
			"chat.model: manifest key %q must be a mapping, got %T",
			key,
			value,
		)
	}

	return child, true, nil
}

func optionalString(
	mapping map[string]any,
	keys ...string,
) (string, bool, error) {
	value, ok := firstValue(mapping, keys...)

	if !ok || value == nil {
		return "", false, nil
	}

	text, ok := value.(string)

	if !ok {
		return "", false, fmt.Errorf(
			"chat.model: manifest key %q must be a string, got %T",
			keys[0],
			value,
		)
	}

	return strings.TrimSpace(text), true, nil
}

func optionalInt64(
	mapping map[string]any,
	key string,
) (int64, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return 0, false, nil
	}

	integer, err := int64FromAny(value)

	if err != nil {
		return 0, false, fmt.Errorf("chat.model: manifest key %q: %w", key, err)
	}

	return integer, true, nil
}

func int64FromAny(value any) (int64, error) {
	switch typed := value.(type) {
	case int:
		return int64(typed), nil
	case int64:
		return typed, nil
	case uint64:
		return int64(typed), nil
	case float64:
		if typed != float64(int64(typed)) {
			return 0, fmt.Errorf("expected integer, got %v", typed)
		}

		return int64(typed), nil
	case string:
		return strconv.ParseInt(strings.TrimSpace(typed), 10, 64)
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func firstValue(mapping map[string]any, keys ...string) (any, bool) {
	for _, key := range keys {
		value, ok := mapping[key]

		if ok {
			return value, true
		}
	}

	return nil, false
}

func firstText(values ...string) string {
	for _, value := range values {
		text := strings.TrimSpace(value)

		if text != "" {
			return text
		}
	}

	return ""
}
