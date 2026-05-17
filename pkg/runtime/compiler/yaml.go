package compiler

import "fmt"

/*
yamlMap is a typed assertion helper. The manifest parser produces
map[string]any documents; the compiler navigates them defensively
so a mistyped block surfaces a precise error instead of a panic.
*/
func yamlMap(value any, where string) (map[string]any, error) {
	if value == nil {
		return nil, nil
	}

	typed, ok := value.(map[string]any)

	if !ok {
		return nil, fmt.Errorf("compiler: %s must be a mapping, got %T", where, value)
	}

	return typed, nil
}

/*
yamlList expects a list of arbitrary entries.
*/
func yamlList(value any, where string) ([]any, error) {
	if value == nil {
		return nil, nil
	}

	typed, ok := value.([]any)

	if !ok {
		return nil, fmt.Errorf("compiler: %s must be a sequence, got %T", where, value)
	}

	return typed, nil
}

/*
yamlString expects a scalar string.
*/
func yamlString(value any, where string) (string, error) {
	if value == nil {
		return "", nil
	}

	typed, ok := value.(string)

	if !ok {
		return "", fmt.Errorf("compiler: %s must be a string, got %T", where, value)
	}

	return typed, nil
}

/*
yamlPath walks a map by dotted path and returns the leaf value or nil.
This is used to dig system.runtime out of a top-level document.
*/
func yamlPath(document map[string]any, path ...string) any {
	cursor := any(document)

	for _, segment := range path {
		typed, ok := cursor.(map[string]any)

		if !ok {
			return nil
		}

		value, ok := typed[segment]

		if !ok {
			return nil
		}

		cursor = value
	}

	return cursor
}
