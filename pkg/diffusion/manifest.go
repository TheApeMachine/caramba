package diffusion

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

func ResolveManifestConfig(config Config) (Config, map[string]any, string, error) {
	document, manifestPath, err := parseManifestDocument(config.Manifest)

	if err != nil {
		return config, nil, manifestPath, err
	}

	config.Manifest = manifestPath
	applyDefaults(&config)

	runtimeBlock, ok, err := optionalMap(document, "system", "runtime")

	if err != nil || !ok {
		return config, document, manifestPath, err
	}

	if err := applyRuntime(&config, runtimeBlock); err != nil {
		return config, document, manifestPath, err
	}

	if strings.TrimSpace(config.Output) != "" {
		config.Generation.Output = strings.TrimSpace(config.Output)
	}

	return config, document, manifestPath, nil
}

func CompileManifest(path string) (*manifest.Graph, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultManifest
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
		return nil, path, fmt.Errorf("diffusion: manifest %s: %w", path, err)
	}

	graph, err = manifest.NewCompiler(".").CompileBytes(data)

	if err != nil {
		return nil, path, err
	}

	return graph, path, nil
}

func parseManifestDocument(path string) (map[string]any, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultManifest
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
		return nil, path, fmt.Errorf("diffusion: manifest %s: %w", path, err)
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
		return nil, fmt.Errorf("diffusion: manifest %s is a directory", path)
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
		return nil, fmt.Errorf("diffusion: manifest %s is a directory", path)
	}

	if filepath.IsAbs(path) {
		return manifest.NewCompiler(filepath.Dir(path)).Compile(filepath.Base(path))
	}

	return manifest.NewCompiler(".").Compile(path)
}

func applyDefaults(config *Config) {
	config.Runtime = "diffusion"
	config.Backend = "auto"
	config.Generation.Height = 1024
	config.Generation.Width = 1024
	config.Generation.LatentChannels = 128
	config.Generation.LatentDownsample = 16
	config.Generation.MaxSequenceLength = 1024
	config.Generation.Seed = 1337
	config.Generation.Output = "caramba-image.png"
	config.Generation.PadTokenID = 151643
	config.Scheduler.Type = "flow_match_euler_discrete"
	config.Scheduler.Steps = 4
	config.Scheduler.NumTrainTimesteps = 1000
	config.Scheduler.BaseImageSeqLen = 256
	config.Scheduler.MaxImageSeqLen = 4096
	config.Scheduler.BaseShift = 0.5
	config.Scheduler.MaxShift = 1.15
	config.Scheduler.Shift = 3
	config.Scheduler.UseDynamicShift = true
	config.Scheduler.TimeShiftType = "exponential"
	config.Model.RepoType = "model"
	config.Tokenizer.RepoType = "model"
	config.TextEncoder.RepoType = "model"
	config.Transformer.RepoType = "model"
	config.VAE.RepoType = "model"
}

func applyRuntime(config *Config, runtimeBlock map[string]any) error {
	if value, ok, err := optionalString(runtimeBlock, "type"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Runtime = value
	}

	if value, ok, err := optionalString(runtimeBlock, "backend"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Backend = value
	}

	if err := applySource(runtimeBlock, "model", &config.Model); err != nil {
		return err
	}

	if err := applySource(runtimeBlock, "tokenizer", &config.Tokenizer); err != nil {
		return err
	}

	if err := applySource(runtimeBlock, "text_encoder", &config.TextEncoder); err != nil {
		return err
	}

	if err := applySource(runtimeBlock, "transformer", &config.Transformer); err != nil {
		return err
	}

	if err := applySource(runtimeBlock, "vae", &config.VAE); err != nil {
		return err
	}

	inheritRuntimeSources(config)

	if generationBlock, ok, err := optionalMap(runtimeBlock, "generation"); ok || err != nil {
		if err != nil {
			return err
		}

		if err := applyGeneration(config, generationBlock); err != nil {
			return err
		}
	}

	if schedulerBlock, ok, err := optionalMap(runtimeBlock, "scheduler"); ok || err != nil {
		if err != nil {
			return err
		}

		if err := applyScheduler(config, schedulerBlock); err != nil {
			return err
		}
	}

	return nil
}

func applySource(mapping map[string]any, key string, source *Source) error {
	manifestSource, ok, err := optionalSource(mapping, key)

	if err != nil || !ok {
		return err
	}

	*source = manifestSource

	if source.RepoType == "" {
		source.RepoType = "model"
	}

	return nil
}

func inheritRuntimeSources(config *Config) {
	if config.Tokenizer.Source == "" {
		config.Tokenizer = config.Model
	}

	config.TextEncoder = inheritSource(config.Model, config.TextEncoder)
	config.Transformer = inheritSource(config.Model, config.Transformer)
	config.VAE = inheritSource(config.Model, config.VAE)
}

func inheritSource(parent Source, child Source) Source {
	if child.Source == "" {
		child.Source = parent.Source
	}

	if child.Cache == "" {
		child.Cache = parent.Cache
	}

	if child.Revision == "" {
		child.Revision = parent.Revision
	}

	if child.RepoType == "" {
		child.RepoType = parent.RepoType
	}

	return child
}

func applyGeneration(config *Config, generationBlock map[string]any) error {
	if value, ok, err := optionalInt(generationBlock, "height"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Height = value
	}

	if value, ok, err := optionalInt(generationBlock, "width"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Width = value
	}

	if value, ok, err := optionalInt(generationBlock, "latent_channels"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.LatentChannels = value
	}

	if value, ok, err := optionalInt(generationBlock, "latent_downsample"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.LatentDownsample = value
	}

	if value, ok, err := optionalInt(generationBlock, "max_sequence_length"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.MaxSequenceLength = value
	}

	if value, ok, err := optionalInt(generationBlock, "pad_token_id"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.PadTokenID = value
	}

	if value, ok, err := optionalInt64(generationBlock, "seed"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Seed = value
	}

	if value, ok, err := optionalString(generationBlock, "output"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.Output = value
	}

	if value, ok, err := optionalRawString(generationBlock, "prompt_template"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Generation.PromptTemplate = value
	}

	return nil
}

func applyScheduler(config *Config, schedulerBlock map[string]any) error {
	if value, ok, err := optionalString(schedulerBlock, "type"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Scheduler.Type = value
	}

	if value, ok, err := optionalInt(schedulerBlock, "num_inference_steps"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Scheduler.Steps = value
	}

	return nil
}

func optionalSource(mapping map[string]any, key string) (Source, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return Source{}, false, nil
	}

	switch typed := value.(type) {
	case string:
		return Source{Source: strings.TrimSpace(typed), RepoType: "model"}, true, nil
	case map[string]any:
		return sourceFromMap(typed)
	default:
		return Source{}, false, fmt.Errorf(
			"diffusion: runtime.%s must be a string or mapping, got %T",
			key,
			value,
		)
	}
}

func sourceFromMap(mapping map[string]any) (Source, bool, error) {
	source, _, err := optionalString(mapping, "source", "repo", "id", "path")

	if err != nil {
		return Source{}, false, err
	}

	file, _, err := optionalString(mapping, "file")

	if err != nil {
		return Source{}, false, err
	}

	cache, _, err := optionalString(mapping, "cache")

	if err != nil {
		return Source{}, false, err
	}

	revision, _, err := optionalString(mapping, "revision")

	if err != nil {
		return Source{}, false, err
	}

	repoType, _, err := optionalString(mapping, "repo_type", "repoType")

	if err != nil {
		return Source{}, false, err
	}

	manifestPath, _, err := optionalString(mapping, "manifest")

	if err != nil {
		return Source{}, false, err
	}

	return Source{
		Source:   source,
		File:     file,
		Cache:    cache,
		Revision: revision,
		RepoType: repoType,
		Manifest: manifestPath,
	}, true, nil
}

func optionalMap(mapping map[string]any, keys ...string) (map[string]any, bool, error) {
	var current any = mapping

	for _, key := range keys {
		parent, ok := current.(map[string]any)

		if !ok {
			return nil, false, fmt.Errorf("diffusion: manifest key %q parent must be a mapping", key)
		}

		value, ok := parent[key]

		if !ok || value == nil {
			return nil, false, nil
		}

		current = value
	}

	out, ok := current.(map[string]any)

	if !ok {
		return nil, false, fmt.Errorf(
			"diffusion: manifest key %q must be a mapping, got %T",
			keys[len(keys)-1],
			current,
		)
	}

	return out, true, nil
}

func optionalString(mapping map[string]any, keys ...string) (string, bool, error) {
	value, ok := firstValue(mapping, keys...)

	if !ok || value == nil {
		return "", false, nil
	}

	text, ok := value.(string)

	if !ok {
		return "", false, fmt.Errorf(
			"diffusion: manifest key %q must be a string, got %T",
			keys[0],
			value,
		)
	}

	return strings.TrimSpace(text), true, nil
}

func optionalRawString(mapping map[string]any, keys ...string) (string, bool, error) {
	value, ok := firstValue(mapping, keys...)

	if !ok || value == nil {
		return "", false, nil
	}

	text, ok := value.(string)

	if !ok {
		return "", false, fmt.Errorf(
			"diffusion: manifest key %q must be a string, got %T",
			keys[0],
			value,
		)
	}

	return text, true, nil
}

func optionalInt(mapping map[string]any, key string) (int, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return 0, false, nil
	}

	integer, err := intFromAny(value)

	if err != nil {
		return 0, false, fmt.Errorf("diffusion: manifest key %q: %w", key, err)
	}

	return integer, true, nil
}

func optionalInt64(mapping map[string]any, key string) (int64, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return 0, false, nil
	}

	integer, err := int64FromAny(value)

	if err != nil {
		return 0, false, fmt.Errorf("diffusion: manifest key %q: %w", key, err)
	}

	return integer, true, nil
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

func intFromAny(value any) (int, error) {
	integer, err := int64FromAny(value)

	if err != nil {
		return 0, err
	}

	return int(integer), nil
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
	default:
		return 0, fmt.Errorf("expected integer, got %T", value)
	}
}
