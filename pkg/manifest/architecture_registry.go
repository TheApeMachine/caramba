package manifest

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
)

const defaultArchitectureRegistry = "model.architecture.registry"

type architectureRegistryEntry struct {
	include   string
	variables map[string]any
}

func (ctx *parseContext) loadArchitectureEntry(
	registryPath string,
	architecture string,
) (architectureRegistryEntry, error) {
	if strings.TrimSpace(registryPath) == "" {
		registryPath = defaultArchitectureRegistry
	}

	registry, err := ctx.loadArchitectureRegistry(registryPath)

	if err != nil {
		return architectureRegistryEntry{}, err
	}

	rawEntry, ok := registry[architecture]

	if !ok {
		return architectureRegistryEntry{}, fmt.Errorf(
			"manifest: architecture %q is not registered in %s",
			architecture,
			registryPath,
		)
	}

	entry, ok := rawEntry.(map[string]any)

	if !ok {
		return architectureRegistryEntry{}, fmt.Errorf(
			"manifest: architecture %q registry entry must be a mapping, got %T",
			architecture,
			rawEntry,
		)
	}

	include, ok := entry["include"].(string)

	if !ok || strings.TrimSpace(include) == "" {
		return architectureRegistryEntry{}, fmt.Errorf(
			"manifest: architecture %q registry entry needs include",
			architecture,
		)
	}

	variables, ok := entry["variables"].(map[string]any)

	if !ok {
		return architectureRegistryEntry{}, fmt.Errorf(
			"manifest: architecture %q registry entry needs variables mapping",
			architecture,
		)
	}

	return architectureRegistryEntry{
		include:   strings.TrimSpace(include),
		variables: variables,
	}, nil
}

func (ctx *parseContext) loadArchitectureRegistry(
	registryPath string,
) (map[string]any, error) {
	relativePath := strings.ReplaceAll(
		registryPath,
		".",
		string(filepath.Separator),
	) + ".yml"

	resolvedPath, err := ctx.parser.resolveRootPath(relativePath)

	if err != nil {
		return nil, err
	}

	registryNode, err := ctx.parser.loadYAMLNode(resolvedPath)

	if err != nil {
		return nil, err
	}

	raw, err := ctx.nodeToAny(registryNode)

	if err != nil {
		return nil, err
	}

	document, ok := raw.(map[string]any)

	if !ok {
		return nil, fmt.Errorf("manifest: architecture registry must be a mapping")
	}

	architectures, ok := document["architectures"].(map[string]any)

	if !ok {
		return nil, fmt.Errorf(
			"manifest: architecture registry %s needs architectures mapping",
			registryPath,
		)
	}

	return architectures, nil
}

func (entry architectureRegistryEntry) resolveVariables(
	config map[string]any,
) (map[string]any, error) {
	out := make(map[string]any, len(entry.variables))

	for name, expression := range entry.variables {
		value, err := evaluateArchitectureExpression(expression, config)

		if err != nil {
			return nil, fmt.Errorf("manifest: architecture variable %s: %w", name, err)
		}

		out[name] = value
	}

	return out, nil
}

func evaluateArchitectureExpression(expression any, config map[string]any) (any, error) {
	mapping, ok := expression.(map[string]any)

	if !ok {
		return expression, nil
	}

	if configKey, ok := mapping["config"].(string); ok {
		value, found := lookupConfigValue(config, configKey)

		if !found {
			return nil, fmt.Errorf("config key %q not found", configKey)
		}

		if number, ok := architectureNumber(value); ok {
			return normalizeArchitectureNumber(number), nil
		}

		return value, nil
	}

	if value, ok := mapping["value"]; ok {
		return value, nil
	}

	if terms, ok := mapping["sum"]; ok {
		return evaluateArchitectureArithmetic(terms, config, arithmeticSum)
	}

	if terms, ok := mapping["product"]; ok {
		return evaluateArchitectureArithmetic(terms, config, arithmeticProduct)
	}

	return nil, fmt.Errorf("unknown expression %v", mapping)
}

func lookupConfigValue(config map[string]any, dotPath string) (any, bool) {
	segments := strings.Split(dotPath, ".")
	var cursor any = config

	for _, segment := range segments {
		mapping, ok := cursor.(map[string]any)

		if !ok {
			return nil, false
		}

		value, ok := mapping[segment]

		if !ok {
			return nil, false
		}

		cursor = value
	}

	return cursor, true
}

type arithmeticMode string

const (
	arithmeticSum     arithmeticMode = "sum"
	arithmeticProduct arithmeticMode = "product"
)

func evaluateArchitectureArithmetic(
	terms any,
	config map[string]any,
	mode arithmeticMode,
) (any, error) {
	sequence, ok := terms.([]any)

	if !ok {
		return nil, fmt.Errorf("%s terms must be a sequence", mode)
	}

	if len(sequence) == 0 {
		return nil, fmt.Errorf("%s needs at least one term", mode)
	}

	total := 0.0

	if mode == arithmeticProduct {
		total = 1.0
	}

	for _, term := range sequence {
		value, err := evaluateArchitectureExpression(term, config)

		if err != nil {
			return nil, err
		}

		number, ok := architectureNumber(value)

		if !ok {
			return nil, fmt.Errorf("%s term must be numeric, got %T", mode, value)
		}

		if mode == arithmeticProduct {
			total *= number
			continue
		}

		total += number
	}

	return normalizeArchitectureNumber(total), nil
}

func architectureNumber(value any) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	default:
		return 0, false
	}
}

func normalizeArchitectureNumber(value float64) any {
	if math.Trunc(value) == value {
		return int(value)
	}

	return value
}
