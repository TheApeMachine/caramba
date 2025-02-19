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
ExtractJSONBlocks finds and parses JSON objects from a string.
It specifically looks for JSON content within markdown-style code blocks
that are marked with the 'json' language identifier. This is particularly
useful when processing unstructured outputs from AI tools. If a JSON array
is found, each object within it will be added to the results individually.
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
Returns nil if the input is not valid JSON, making it safe for parsing
potentially invalid input.
*/
func ParseJSON(s string) map[string]any {
	var result map[string]any
	if err := json.Unmarshal([]byte(s), &result); err == nil {
		return result
	}
	return nil
}
