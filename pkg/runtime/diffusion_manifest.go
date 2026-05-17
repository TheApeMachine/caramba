package runtime

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/manifest"
)

const (
	diffusionManifestPrefix = "runtime/diffusion"
	diffusionRuntimeAsset   = "runtime/diffusion.yml"
)

func ResolveDiffusionConfig(
	config DiffusionConfig,
) (DiffusionConfig, map[string]any, string, error) {
	document, manifestPath, err := parseDiffusionManifestDocument(config.Manifest)

	if err != nil {
		return config, nil, manifestPath, err
	}

	config.Manifest = manifestPath
	applyDiffusionDefaults(&config)

	runtimeBlock, ok, err := optionalMap(document, diffusionManifestPrefix, "system", "runtime")

	if err != nil || !ok {
		return config, document, manifestPath, err
	}

	if err := applyDiffusionRuntime(&config, runtimeBlock); err != nil {
		return config, document, manifestPath, err
	}

	if strings.TrimSpace(config.Output) != "" {
		config.Generation.Output = strings.TrimSpace(config.Output)
	}

	return config, document, manifestPath, nil
}

func CompileDiffusionManifest(path string) (*manifest.Graph, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultDiffusionManifest
	}

	graph, err := compileLocalManifest(path)

	if err == nil {
		return graph, path, nil
	}

	if !errors.Is(err, os.ErrNotExist) {
		return nil, path, err
	}

	graph, err = manifest.NewCompiler(".").WithFS(asset.TemplateFS()).Compile(path)

	if err != nil {
		return nil, path, fmt.Errorf("runtime/diffusion: manifest %s: %w", path, err)
	}

	return graph, path, nil
}

func parseDiffusionManifestDocument(path string) (map[string]any, string, error) {
	path = strings.TrimSpace(path)

	if path == "" {
		path = DefaultDiffusionManifest
	}

	document, err := parseLocalManifestDocument(path)

	if err == nil {
		return document, path, nil
	}

	if !errors.Is(err, os.ErrNotExist) {
		return nil, path, err
	}

	document, err = manifest.NewParser(".").WithFS(asset.TemplateFS()).Parse(path)

	if err != nil {
		return nil, path, fmt.Errorf("runtime/diffusion: manifest %s: %w", path, err)
	}

	return document, path, nil
}

func applyDiffusionDefaults(config *DiffusionConfig) {
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

func applyDiffusionRuntime(config *DiffusionConfig, runtimeBlock map[string]any) error {
	if value, ok, err := optionalString(runtimeBlock, diffusionManifestPrefix, "type"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Runtime = value
	}

	if value, ok, err := optionalString(runtimeBlock, diffusionManifestPrefix, "backend"); ok || err != nil {
		if err != nil {
			return err
		}

		config.Backend = value
	}

	if err := applyDiffusionSource(runtimeBlock, "model", &config.Model); err != nil {
		return err
	}

	if err := applyDiffusionSource(runtimeBlock, "tokenizer", &config.Tokenizer); err != nil {
		return err
	}

	if err := applyDiffusionSource(runtimeBlock, "text_encoder", &config.TextEncoder); err != nil {
		return err
	}

	if err := applyDiffusionSource(runtimeBlock, "transformer", &config.Transformer); err != nil {
		return err
	}

	if err := applyDiffusionSource(runtimeBlock, "vae", &config.VAE); err != nil {
		return err
	}

	inheritRuntimeSources(config)

	if generationBlock, ok, err := optionalMap(runtimeBlock, diffusionManifestPrefix, "generation"); ok || err != nil {
		if err != nil {
			return err
		}

		if err := applyGeneration(config, generationBlock); err != nil {
			return err
		}
	}

	if schedulerBlock, ok, err := optionalMap(runtimeBlock, diffusionManifestPrefix, "scheduler"); ok || err != nil {
		if err != nil {
			return err
		}

		if err := applyScheduler(config, schedulerBlock); err != nil {
			return err
		}
	}

	return nil
}

func applyDiffusionSource(mapping map[string]any, key string, source *Source) error {
	manifestSource, ok, err := optionalSource(mapping, key, diffusionManifestPrefix)

	if err != nil || !ok {
		return err
	}

	*source = Source{
		Source:   manifestSource.source,
		File:     manifestSource.file,
		Cache:    manifestSource.cache,
		Revision: manifestSource.revision,
		RepoType: manifestSource.repoType,
		Manifest: manifestSource.manifest,
	}

	if source.RepoType == "" {
		source.RepoType = "model"
	}

	return nil
}

func inheritRuntimeSources(config *DiffusionConfig) {
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
