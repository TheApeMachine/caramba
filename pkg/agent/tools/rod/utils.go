package rod

import (
	"strings"
	"time"
)

// getTimeoutDuration returns a timeout duration from args or the default
func getTimeoutDuration(args map[string]interface{}, defaultTimeout time.Duration) time.Duration {
	if timeout, ok := args["timeout"].(int); ok {
		return time.Duration(timeout) * time.Second
	}
	if timeout, ok := args["timeout"].(float64); ok {
		return time.Duration(timeout) * time.Second
	}
	return defaultTimeout
}

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

// extractBetween extracts text between startStr and endStr
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

// extractBetweenAll extracts all occurrences of text between startStr and endStr
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

// stripTags removes all HTML tags from text
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
