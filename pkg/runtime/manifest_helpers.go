package runtime

import (
	"fmt"
	"strconv"
	"strings"
)

type manifestSource struct {
	source   string
	file     string
	cache    string
	revision string
	repoType string
	manifest string
}

func optionalSource(
	mapping map[string]any,
	key string,
	prefix string,
) (manifestSource, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return manifestSource{}, false, nil
	}

	switch typed := value.(type) {
	case string:
		return manifestSource{source: strings.TrimSpace(typed), repoType: "model"}, true, nil
	case map[string]any:
		return sourceFromMap(typed, prefix)
	default:
		return manifestSource{}, false, fmt.Errorf(
			"%s: runtime.%s must be a string or mapping, got %T",
			prefix,
			key,
			value,
		)
	}
}

func sourceFromMap(mapping map[string]any, prefix string) (manifestSource, bool, error) {
	source, _, err := optionalString(mapping, prefix, "source", "repo", "id", "path")

	if err != nil {
		return manifestSource{}, false, err
	}

	file, _, err := optionalString(mapping, prefix, "file")

	if err != nil {
		return manifestSource{}, false, err
	}

	cache, _, err := optionalString(mapping, prefix, "cache")

	if err != nil {
		return manifestSource{}, false, err
	}

	revision, _, err := optionalString(mapping, prefix, "revision")

	if err != nil {
		return manifestSource{}, false, err
	}

	repoType, _, err := optionalString(mapping, prefix, "repo_type", "repoType")

	if err != nil {
		return manifestSource{}, false, err
	}

	manifestPath, _, err := optionalString(mapping, prefix, "manifest")

	if err != nil {
		return manifestSource{}, false, err
	}

	return manifestSource{
		source:   source,
		file:     file,
		cache:    cache,
		revision: revision,
		repoType: repoType,
		manifest: manifestPath,
	}, true, nil
}

func optionalMap(
	mapping map[string]any,
	prefix string,
	keys ...string,
) (map[string]any, bool, error) {
	var current any = mapping

	for _, key := range keys {
		parent, ok := current.(map[string]any)

		if !ok {
			return nil, false, fmt.Errorf("%s: manifest key %q parent must be a mapping", prefix, key)
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
			"%s: manifest key %q must be a mapping, got %T",
			prefix,
			keys[len(keys)-1],
			current,
		)
	}

	return out, true, nil
}

func optionalString(
	mapping map[string]any,
	prefix string,
	keys ...string,
) (string, bool, error) {
	value, ok := firstValue(mapping, keys...)

	if !ok || value == nil {
		return "", false, nil
	}

	text, ok := value.(string)

	if !ok {
		return "", false, fmt.Errorf(
			"%s: manifest key %q must be a string, got %T",
			prefix,
			keys[0],
			value,
		)
	}

	return strings.TrimSpace(text), true, nil
}

func optionalRawString(
	mapping map[string]any,
	prefix string,
	keys ...string,
) (string, bool, error) {
	value, ok := firstValue(mapping, keys...)

	if !ok || value == nil {
		return "", false, nil
	}

	text, ok := value.(string)

	if !ok {
		return "", false, fmt.Errorf(
			"%s: manifest key %q must be a string, got %T",
			prefix,
			keys[0],
			value,
		)
	}

	return text, true, nil
}

func optionalInt(mapping map[string]any, prefix string, key string) (int, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return 0, false, nil
	}

	integer, err := int64FromAny(value)

	if err != nil {
		return 0, false, fmt.Errorf("%s: manifest key %q: %w", prefix, key, err)
	}

	return int(integer), true, nil
}

func optionalInt64(mapping map[string]any, prefix string, key string) (int64, bool, error) {
	value, ok := mapping[key]

	if !ok || value == nil {
		return 0, false, nil
	}

	integer, err := int64FromAny(value)

	if err != nil {
		return 0, false, fmt.Errorf("%s: manifest key %q: %w", prefix, key, err)
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
