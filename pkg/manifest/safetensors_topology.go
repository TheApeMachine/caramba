package manifest

import (
	"context"
	"fmt"
	"maps"
	"strings"

	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"gopkg.in/yaml.v3"
)

func (ctx *parseContext) extractSafeTensorsTopology(
	yamlNode *yaml.Node,
) (*yaml.Node, bool) {
	if len(yamlNode.Content) != 2 {
		return nil, false
	}

	for pairIndex := 0; pairIndex < len(yamlNode.Content)-1; pairIndex += 2 {
		key := yamlNode.Content[pairIndex].Value

		if key == "from_safetensors" {
			return yamlNode.Content[pairIndex+1], true
		}
	}

	return nil, false
}

func (ctx *parseContext) resolveSafeTensorsTopology(specNode *yaml.Node) (any, error) {
	resolved, err := ctx.resolveNode(specNode)

	if err != nil {
		return nil, err
	}

	spec, ok := resolved.(map[string]any)

	if !ok {
		return nil, fmt.Errorf(
			"manifest: from_safetensors must be a mapping, got %T",
			resolved,
		)
	}

	source, err := safeTensorsSource(spec)

	if err != nil {
		return nil, err
	}

	config, err := safeTensorsConfig(source, spec)

	if err != nil {
		return nil, err
	}

	architecture, err := safeTensorsArchitecture(spec, config)

	if err != nil {
		return nil, err
	}

	registryPath := safeTensorsString(spec, "registry")
	entry, err := ctx.loadArchitectureEntry(registryPath, architecture)

	if err != nil {
		return nil, err
	}

	variables, err := entry.resolveVariables(config)

	if err != nil {
		return nil, err
	}

	overrides, err := safeTensorsVariables(spec)

	if err != nil {
		return nil, err
	}

	maps.Copy(variables, overrides)

	return ctx.loadIncludeTarget(entry.include, variables)
}

func safeTensorsSource(spec map[string]any) (modelweights.Source, error) {
	source := modelweights.Source{
		Source:   safeTensorsString(spec, "source", "repo", "id", "path"),
		File:     safeTensorsString(spec, "file"),
		Cache:    safeTensorsString(spec, "cache"),
		Revision: safeTensorsString(spec, "revision"),
		RepoType: safeTensorsString(spec, "repo_type", "repoType"),
	}

	if source.RepoType == "" {
		source.RepoType = "model"
	}

	if source.Source != "" {
		return source, nil
	}

	if _, ok := spec["config"]; ok {
		return source, nil
	}

	return modelweights.Source{}, fmt.Errorf(
		"manifest: from_safetensors needs source when config is not embedded",
	)
}

func safeTensorsConfig(
	source modelweights.Source,
	spec map[string]any,
) (map[string]any, error) {
	if rawConfig, ok := spec["config"]; ok {
		config, ok := rawConfig.(map[string]any)

		if !ok {
			return nil, fmt.Errorf(
				"manifest: from_safetensors.config must be a mapping, got %T",
				rawConfig,
			)
		}

		return config, nil
	}

	return modelweights.ResolveConfig(context.Background(), source)
}

func safeTensorsArchitecture(
	spec map[string]any,
	config map[string]any,
) (string, error) {
	if architecture := safeTensorsString(spec, "architecture"); architecture != "" {
		return architecture, nil
	}

	if architecture := safeTensorsString(config, "architecture"); architecture != "" {
		return architecture, nil
	}

	rawArchitectures, ok := config["architectures"]

	if !ok {
		return "", fmt.Errorf(
			"manifest: from_safetensors config needs architectures[0]",
		)
	}

	architecture, ok := firstArchitecture(rawArchitectures)

	if !ok {
		return "", fmt.Errorf(
			"manifest: from_safetensors architectures must contain a string",
		)
	}

	return architecture, nil
}

func firstArchitecture(raw any) (string, bool) {
	switch typed := raw.(type) {
	case []any:
		if len(typed) == 0 {
			return "", false
		}

		architecture, ok := typed[0].(string)

		return strings.TrimSpace(architecture), ok && strings.TrimSpace(architecture) != ""
	case []string:
		if len(typed) == 0 {
			return "", false
		}

		return strings.TrimSpace(typed[0]), strings.TrimSpace(typed[0]) != ""
	default:
		return "", false
	}
}

func safeTensorsVariables(spec map[string]any) (map[string]any, error) {
	rawVariables, ok := spec["variables"]

	if !ok || rawVariables == nil {
		return map[string]any{}, nil
	}

	variables, ok := rawVariables.(map[string]any)

	if !ok {
		return nil, fmt.Errorf(
			"manifest: from_safetensors.variables must be a mapping, got %T",
			rawVariables,
		)
	}

	return variables, nil
}

func safeTensorsString(mapping map[string]any, keys ...string) string {
	for _, key := range keys {
		value, ok := mapping[key]

		if !ok {
			continue
		}

		text, ok := value.(string)

		if ok {
			return strings.TrimSpace(text)
		}
	}

	return ""
}
