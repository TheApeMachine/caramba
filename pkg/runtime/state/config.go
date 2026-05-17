package state

import "fmt"

/*
intFromConfig reads a typed integer from a state-config map, treating
missing as zero. Accepts the YAML decoder's native int forms plus
string values produced by ${variable} interpolation.
*/
func intFromConfig(config map[string]any, key string) (int, error) {
	value, ok := config[key]

	if !ok {
		return 0, nil
	}

	parsed, err := intFromAny(value)

	if err != nil {
		return 0, fmt.Errorf("runtime/state: %q: %w", key, err)
	}

	return parsed, nil
}

/*
int64FromConfig is the int64-flavoured variant used for seeds.
*/
func int64FromConfig(config map[string]any, key string) (int64, error) {
	value, ok := config[key]

	if !ok {
		return 0, nil
	}

	parsed, err := intFromAny(value)

	if err != nil {
		return 0, fmt.Errorf("runtime/state: %q: %w", key, err)
	}

	return int64(parsed), nil
}

/*
stringFromConfig reads an optional string from the configuration.
*/
func stringFromConfig(config map[string]any, key string) (string, error) {
	value, ok := config[key]

	if !ok {
		return "", nil
	}

	typed, ok := value.(string)

	if !ok {
		return "", fmt.Errorf("runtime/state: %q must be a string, got %T", key, value)
	}

	return typed, nil
}

/*
intSliceFromConfig reads an optional []int field from the
configuration. Used for tensor shapes.
*/
func intSliceFromConfig(config map[string]any, key string) ([]int, error) {
	value, ok := config[key]

	if !ok {
		return nil, nil
	}

	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		out := make([]int, len(typed))

		for index, entry := range typed {
			cast, err := intFromAny(entry)

			if err != nil {
				return nil, fmt.Errorf("runtime/state: %q[%d]: %w", key, index, err)
			}

			out[index] = cast
		}

		return out, nil
	}

	return nil, fmt.Errorf("runtime/state: %q must be a list of integers, got %T", key, value)
}

func intFromAny(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int64:
		return int(typed), nil
	case int32:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case string:
		parsed, err := parseIntString(typed)

		if err != nil {
			return 0, fmt.Errorf("expected integer, got string %q", typed)
		}

		return parsed, nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func parseIntString(text string) (int, error) {
	value := 0
	sign := 1
	start := 0

	if len(text) == 0 {
		return 0, fmt.Errorf("empty string")
	}

	if text[0] == '-' {
		sign = -1
		start = 1
	}

	if start == len(text) {
		return 0, fmt.Errorf("only a sign")
	}

	for index := start; index < len(text); index++ {
		character := text[index]

		if character < '0' || character > '9' {
			return 0, fmt.Errorf("non-digit %q", character)
		}

		value = value*10 + int(character-'0')
	}

	return sign * value, nil
}
