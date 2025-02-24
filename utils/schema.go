package utils

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/invopop/jsonschema"
)

/*
GenerateSchema generates a JSON schema for a given type.
*/
func GenerateSchema[T any]() interface{} {
	// Structured Outputs uses a subset of JSON schema
	// These flags are necessary to comply with the subset
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}

/*
ExtractJSON finds and parses JSON objects from a string.
First tries to extract markdown-style JSON blocks, then falls back to finding raw JSON objects.
Returns a slice of successfully parsed JSON objects.
*/
func ExtractJSON(s string) []map[string]any {
	// First try markdown blocks
	if blocks := ExtractJSONBlocks(s); len(blocks) > 0 {
		return blocks
	}

	// Fallback to finding raw JSON objects
	var results []map[string]any
	var start int
	var depth int

	// Track nested objects by counting brace depth
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth == 0 {
				// Only try to parse when we've found a complete top-level object
				if obj := ParseJSON(s[start : i+1]); obj != nil {
					results = append(results, obj)
				}
			}
		}
	}

	return results
}

/*
ExtractJSONBlocks extracts and parses JSON from markdown code blocks.
*/
func ExtractJSONBlocks(s string) []map[string]any {
	// Extract blocks marked with json language identifier
	re := regexp.MustCompile("```json([\\s\\S]*?)```")
	matches := re.FindAllStringSubmatch(s, -1)

	var results []map[string]any
	for _, match := range matches {
		if len(match) >= 2 {
			content := strings.TrimSpace(match[1])
			// Try parsing as array first
			var arrayResult []map[string]any
			if err := json.Unmarshal([]byte(content), &arrayResult); err == nil {
				results = append(results, arrayResult...)
				continue
			}

			// If not an array, try parsing as single object
			if block := ParseJSON(content); block != nil {
				results = append(results, block)
			}
		}
	}

	return results
}

/*
ParseJSON safely converts a JSON string into a map.
Returns nil if the input is not valid JSON.
*/
func ParseJSON(s string) map[string]any {
	var result map[string]any
	if err := json.Unmarshal([]byte(s), &result); err == nil {
		return result
	}
	return nil
}
