package rod

import (
	"strings"
	"time"
)

func getTimeoutDuration(args map[string]interface{}, defaultTimeout time.Duration) time.Duration {
	if timeout, ok := args["timeout"].(int); ok {
		return time.Duration(timeout) * time.Second
	}

	if timeout, ok := args["timeout"].(float64); ok {
		return time.Duration(timeout) * time.Second
	}

	return defaultTimeout
}

func getStringOr(args map[string]interface{}, key, defaultValue string) string {
	if val, ok := args[key].(string); ok && val != "" {
		return val
	}

	return defaultValue
}

func getBoolOr(args map[string]interface{}, key string, defaultValue bool) bool {
	if val, ok := args[key].(bool); ok {
		return val
	}

	return defaultValue
}

func extractBetween(text, startStr, endStr string) string {
	startIdx := strings.Index(text, startStr)

	if startIdx == -1 {
		return ""
	}

	startIdx += len(startStr)
	endIdx := strings.Index(text[startIdx:], endStr)

	if endIdx == -1 {
		return ""
	}

	return text[startIdx : startIdx+endIdx]
}

func extractBetweenAll(text, startStr, endStr string) []string {
	var results []string

	for {
		startIdx := strings.Index(text, startStr)

		if startIdx == -1 {
			break
		}

		startIdx += len(startStr)
		endIdx := strings.Index(text[startIdx:], endStr)

		if endIdx == -1 {
			break
		}

		results = append(results, text[startIdx:startIdx+endIdx])
		text = text[startIdx+endIdx+len(endStr):]
	}

	return results
}

func stripTags(text string) string {
	var result strings.Builder
	var inTag bool

	for _, r := range text {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
		case !inTag:
			result.WriteRune(r)
		}
	}

	return result.String()
}
