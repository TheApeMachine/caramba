package browser

import (
	"fmt"
	"time"
)

// getStringOr returns the string value from args with the given key, or the default value if not found
func getStringOr(args map[string]interface{}, key, defaultValue string) string {
	if val, ok := args[key].(string); ok && val != "" {
		return val
	}
	return defaultValue
}

// getBoolOr returns the boolean value from args with the given key, or the default value if not found
func getBoolOr(args map[string]interface{}, key string, defaultValue bool) bool {
	if val, ok := args[key].(bool); ok {
		return val
	}
	return defaultValue
}

// getIntOr returns the int value from args with the given key, or the default value if not found
func getIntOr(args map[string]interface{}, key string, defaultValue int) int {
	switch v := args[key].(type) {
	case int:
		return v
	case float64:
		return int(v)
	default:
		return defaultValue
	}
}

// getFloatOr returns the float64 value from args with the given key, or the default value if not found
func getFloatOr(args map[string]interface{}, key string, defaultValue float64) float64 {
	switch v := args[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return defaultValue
	}
}

// getDurationOr returns the duration value from args with the given key, or the default value if not found
func getDurationOr(args map[string]interface{}, key string, defaultValue time.Duration) time.Duration {
	switch v := args[key].(type) {
	case string:
		if duration, err := time.ParseDuration(v); err == nil {
			return duration
		}
		return defaultValue
	case float64:
		return time.Duration(v) * time.Millisecond
	case int:
		return time.Duration(v) * time.Millisecond
	default:
		return defaultValue
	}
}

// formatError formats an error with additional context
func formatError(baseErr error, context string) error {
	if baseErr == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", context, baseErr)
}
