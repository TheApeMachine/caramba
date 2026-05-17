package scheduler

import (
	"fmt"
	"sort"
)

type flowMatchConfig struct {
	steps             int
	numTrainTimesteps int
}

func parseFlowMatchConfig(raw map[string]any) (flowMatchConfig, error) {
	configuration := flowMatchConfig{numTrainTimesteps: 1000}

	steps, err := intFromConfig(raw, "steps")

	if err != nil {
		return flowMatchConfig{}, err
	}

	if steps <= 0 {
		return flowMatchConfig{}, fmt.Errorf("scheduler/flow_match: steps must be positive")
	}

	configuration.steps = steps

	numTrainTimesteps, err := intFromConfig(raw, "num_train_timesteps")

	if err != nil {
		return flowMatchConfig{}, err
	}

	if numTrainTimesteps > 0 {
		configuration.numTrainTimesteps = numTrainTimesteps
	}

	return configuration, nil
}

func buildSigmas(configuration flowMatchConfig) []float64 {
	if configuration.steps == 1 {
		return []float64{1, 0}
	}

	sigmas := make([]float64, configuration.steps)
	lastIndex := float64(configuration.steps - 1)
	minimum := 1.0 / float64(configuration.steps)

	for index := range sigmas {
		fraction := float64(index) / lastIndex
		sigmas[index] = 1 - fraction*(1-minimum)
	}

	return append(sigmas, 0)
}

func fingerprintConfig(raw map[string]any) string {
	if raw == nil {
		return "{}"
	}

	keys := make([]string, 0, len(raw))

	for key := range raw {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	output := ""

	for _, key := range keys {
		output += fmt.Sprintf("%s=%v;", key, raw[key])
	}

	return output
}

func intFromConfig(raw map[string]any, key string) (int, error) {
	value, ok := raw[key]

	if !ok {
		return 0, nil
	}

	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	case string:
		parsed, err := parseIntString(typed)

		if err != nil {
			return 0, fmt.Errorf("scheduler: %q: %w", key, err)
		}

		return parsed, nil
	}

	return 0, fmt.Errorf("scheduler: %q must be integer, got %T", key, value)
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
