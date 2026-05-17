package value

import "fmt"

/*
appendAny appends one element onto a typed slice.
*/
func appendAny(current any, element any) (any, error) {
	switch typed := current.(type) {
	case []any:
		return append(typed, element), nil
	case []int:
		intElement, err := asInt(element)

		if err != nil {
			return nil, fmt.Errorf("value.append: %w", err)
		}

		return append(typed, intElement), nil
	case []float64:
		floatElement, err := asFloat64(element)

		if err != nil {
			return nil, fmt.Errorf("value.append: %w", err)
		}

		return append(typed, floatElement), nil
	case []string:
		stringElement, ok := element.(string)

		if !ok {
			return nil, fmt.Errorf("value.append: expected string element, got %T", element)
		}

		return append(typed, stringElement), nil
	case nil:
		return []any{element}, nil
	}

	return nil, fmt.Errorf("value.append: unsupported target type %T", current)
}

/*
sliceAny returns source[start:end] respecting the original element
type.
*/
func sliceAny(source any, start int, end int) (any, error) {
	switch typed := source.(type) {
	case []any:
		if err := checkBounds(len(typed), start, end); err != nil {
			return nil, err
		}

		return append([]any(nil), typed[start:end]...), nil
	case []int:
		if err := checkBounds(len(typed), start, end); err != nil {
			return nil, err
		}

		return append([]int(nil), typed[start:end]...), nil
	case []float64:
		if err := checkBounds(len(typed), start, end); err != nil {
			return nil, err
		}

		return append([]float64(nil), typed[start:end]...), nil
	case []string:
		if err := checkBounds(len(typed), start, end); err != nil {
			return nil, err
		}

		return append([]string(nil), typed[start:end]...), nil
	}

	return nil, fmt.Errorf("value.slice: unsupported source type %T", source)
}

func checkBounds(length int, start int, end int) error {
	if start < 0 || end < 0 {
		return fmt.Errorf("value.slice: negative bounds [%d:%d]", start, end)
	}

	if start > end {
		return fmt.Errorf("value.slice: start %d > end %d", start, end)
	}

	if end > length {
		return fmt.Errorf("value.slice: end %d > length %d", end, length)
	}

	return nil
}

func asInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func asFloat64(value any) (float64, error) {
	switch typed := value.(type) {
	case float64:
		return typed, nil
	case float32:
		return float64(typed), nil
	case int:
		return float64(typed), nil
	case int64:
		return float64(typed), nil
	}

	return 0, fmt.Errorf("expected float, got %T", value)
}

func intFromConfig(config map[string]any, key string) (int, error) {
	value, ok := config[key]

	if !ok {
		return 0, nil
	}

	return asInt(value)
}

func intFromConfigOrLen(config map[string]any, key string, fallback any) (int, error) {
	if _, ok := config[key]; ok {
		return intFromConfig(config, key)
	}

	switch typed := fallback.(type) {
	case []any:
		return len(typed), nil
	case []int:
		return len(typed), nil
	case []float64:
		return len(typed), nil
	case []string:
		return len(typed), nil
	}

	return 0, fmt.Errorf("cannot determine length of %T", fallback)
}
