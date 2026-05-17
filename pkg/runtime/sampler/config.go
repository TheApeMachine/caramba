package sampler

import "fmt"

type config struct {
	temperature       float64
	topK              int
	topP              float64
	repetitionPenalty float64
	stopTokenIDs      map[int]bool
	stopSuffixes      [][]int
}

func parseConfig(raw map[string]any) (config, error) {
	configuration := config{temperature: 1.0, repetitionPenalty: 1.0}

	if value, ok := raw["temperature"]; ok {
		temperature, err := asFloat64(value)

		if err != nil {
			return config{}, fmt.Errorf("temperature: %w", err)
		}

		if temperature <= 0 {
			return config{}, fmt.Errorf("temperature must be positive, got %g", temperature)
		}

		configuration.temperature = temperature
	}

	if value, ok := raw["repetition_penalty"]; ok {
		penalty, err := asFloat64(value)

		if err != nil {
			return config{}, fmt.Errorf("repetition_penalty: %w", err)
		}

		if penalty < 1 {
			return config{}, fmt.Errorf(
				"repetition_penalty must be >= 1, got %g", penalty,
			)
		}

		configuration.repetitionPenalty = penalty
	}

	if value, ok := raw["top_k"]; ok {
		topK, err := asInt(value)

		if err != nil {
			return config{}, fmt.Errorf("top_k: %w", err)
		}

		configuration.topK = topK
	}

	if value, ok := raw["top_p"]; ok {
		topP, err := asFloat64(value)

		if err != nil {
			return config{}, fmt.Errorf("top_p: %w", err)
		}

		if topP < 0 || topP > 1 {
			return config{}, fmt.Errorf("top_p must be in [0, 1], got %g", topP)
		}

		configuration.topP = topP
	}

	stopIDs, err := asIntSlice(raw["stop_token_ids"])

	if err != nil {
		return config{}, fmt.Errorf("stop_token_ids: %w", err)
	}

	if len(stopIDs) > 0 {
		configuration.stopTokenIDs = map[int]bool{}

		for _, id := range stopIDs {
			configuration.stopTokenIDs[id] = true
		}
	}

	stopSuffixes, err := asIntMatrix(raw["stop_sequences"])

	if err != nil {
		return config{}, fmt.Errorf("stop_sequences: %w", err)
	}

	configuration.stopSuffixes = stopSuffixes

	return configuration, nil
}

func (configuration config) matchesStop(token int) bool {
	if configuration.stopTokenIDs == nil {
		return false
	}

	return configuration.stopTokenIDs[token]
}

func (configuration config) matchesStopSuffix(history []int, token int) bool {
	if len(configuration.stopSuffixes) == 0 {
		return false
	}

	extended := append(append([]int(nil), history...), token)

	for _, suffix := range configuration.stopSuffixes {
		if len(suffix) == 0 || len(extended) < len(suffix) {
			continue
		}

		tail := extended[len(extended)-len(suffix):]
		matched := true

		for index, value := range suffix {
			if tail[index] != value {
				matched = false

				break
			}
		}

		if matched {
			return true
		}
	}

	return false
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

func asIntSlice(value any) ([]int, error) {
	if value == nil {
		return nil, nil
	}

	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		out := make([]int, len(typed))

		for index, element := range typed {
			cast, err := asInt(element)

			if err != nil {
				return nil, fmt.Errorf("element %d: %w", index, err)
			}

			out[index] = cast
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected []int, got %T", value)
}

func asIntMatrix(value any) ([][]int, error) {
	if value == nil {
		return nil, nil
	}

	switch typed := value.(type) {
	case [][]int:
		out := make([][]int, len(typed))

		for index, row := range typed {
			out[index] = append([]int(nil), row...)
		}

		return out, nil
	case []any:
		out := make([][]int, len(typed))

		for index, element := range typed {
			row, err := asIntSlice(element)

			if err != nil {
				return nil, fmt.Errorf("row %d: %w", index, err)
			}

			out[index] = row
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected [][]int, got %T", value)
}
