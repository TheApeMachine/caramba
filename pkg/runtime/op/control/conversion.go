package control

import (
	"fmt"
	"strconv"

	"github.com/theapemachine/caramba/pkg/runtime/op"
)

/*
resolveCount returns the iteration count for LoopCount. The count
may be a literal int in Config["count"] or a value reference under
Inputs["count"]. Inputs win when both are present.
*/
func resolveCount(execContext op.Context) (int, error) {
	step := execContext.Step()

	if ref, ok := step.Inputs["count"]; ok {
		value, err := execContext.Resolve(ref)

		if err != nil {
			return 0, err
		}

		return asInt(value)
	}

	raw, ok := step.Config["count"]

	if !ok {
		return 0, fmt.Errorf("control.loop_count: config.count or inputs.count is required")
	}

	return asInt(raw)
}

// asInt coerces a runtime value into an int. Float inputs are
// truncated toward zero (matching pkg/runtime/sampler/config.go's
// asInt); callers that need strict integer semantics should validate
// upstream before passing a float through here.
func asInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case uint:
		return int(typed), nil
	case uint32:
		return int(typed), nil
	case uint64:
		return int(typed), nil
	case float32:
		return int(typed), nil
	case float64:
		return int(typed), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func intFromConfig(config map[string]any, key string) (int, error) {
	value, ok := config[key]

	if !ok {
		return 0, nil
	}

	return asInt(value)
}

func asBool(value any) (bool, error) {
	switch typed := value.(type) {
	case bool:
		return typed, nil
	case int:
		return typed != 0, nil
	case int8:
		return typed != 0, nil
	case int16:
		return typed != 0, nil
	case int32:
		return typed != 0, nil
	case int64:
		return typed != 0, nil
	case uint:
		return typed != 0, nil
	case uint8:
		return typed != 0, nil
	case uint16:
		return typed != 0, nil
	case uint32:
		return typed != 0, nil
	case uint64:
		return typed != 0, nil
	case uintptr:
		return typed != 0, nil
	case float32:
		return typed != 0, nil
	case float64:
		return typed != 0, nil
	case string:
		parsed, err := strconv.ParseBool(typed)

		if err != nil {
			return false, fmt.Errorf("expected boolean, got string %q", typed)
		}

		return parsed, nil
	}

	return false, fmt.Errorf("expected boolean, got %T", value)
}

func toAnySlice(value any) ([]any, error) {
	switch typed := value.(type) {
	case []any:
		return typed, nil
	case []int:
		out := make([]any, len(typed))

		for index, element := range typed {
			out[index] = element
		}

		return out, nil
	case []float64:
		out := make([]any, len(typed))

		for index, element := range typed {
			out[index] = element
		}

		return out, nil
	case []string:
		out := make([]any, len(typed))

		for index, element := range typed {
			out[index] = element
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected slice, got %T", value)
}
