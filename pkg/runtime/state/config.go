package state

import (
	"fmt"
	"math"
	"strconv"
)

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
	case int8:
		return int(typed), nil
	case int16:
		return int(typed), nil
	case int32:
		return int(typed), nil
	case int64:
		if typed > int64(maxInt) || typed < int64(minInt) {
			return 0, fmt.Errorf("expected integer, got int64 %d out of int range", typed)
		}

		return int(typed), nil
	case uint:
		if uint64(typed) > uint64(maxInt) {
			return 0, fmt.Errorf("expected integer, got uint %d out of int range", typed)
		}

		return int(typed), nil
	case uint8:
		return int(typed), nil
	case uint16:
		return int(typed), nil
	case uint32:
		return int(typed), nil
	case uint64:
		if typed > uint64(maxInt) {
			return 0, fmt.Errorf("expected integer, got uint64 %d out of int range", typed)
		}

		return int(typed), nil
	case float64:
		if math.Mod(typed, 1.0) != 0 {
			return 0, fmt.Errorf("expected integer, got fractional float %g", typed)
		}

		if typed > float64(maxInt) || typed < float64(minInt) {
			return 0, fmt.Errorf("expected integer, got float64 %g out of int range", typed)
		}

		return int(typed), nil
	case string:
		parsed, err := parseIntString(typed)

		if err != nil {
			return 0, fmt.Errorf("expected integer, got string %q: %w", typed, err)
		}

		return parsed, nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

const (
	maxInt = int(^uint(0) >> 1)
	minInt = -maxInt - 1
)

// parseIntString delegates to strconv.Atoi so overflow and malformed
// input surface as the standard library's *strconv.NumError, including
// values that exceed int range on the running platform.
func parseIntString(text string) (int, error) {
	return strconv.Atoi(text)
}
