package memory

import (
	"encoding/json"
	"fmt"
	"strings"
)

// StringToMapList converts a string to a map[string]interface{} list.
// It parses the input string as JSON and returns a slice of maps.
//
// Parameters:
//   - input: The JSON string to parse
//
// Returns:
//   - A slice of map[string]interface{} representing the parsed JSON
//   - An error if the parsing fails, or nil on success
func StringToMapList(input string) ([]map[string]interface{}, error) {
	var result []map[string]interface{}
	err := json.Unmarshal([]byte(input), &result)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}
	return result, nil
}

// MapToString converts a map to a JSON string.
// It marshals the input map to a JSON string.
//
// Parameters:
//   - input: The map to convert to JSON
//
// Returns:
//   - A JSON string representation of the map
//   - An error if the marshaling fails, or nil on success
func MapToString(input interface{}) (string, error) {
	result, err := json.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("failed to marshal to JSON: %w", err)
	}
	return string(result), nil
}

// InferRelationshipType infers the relationship type from source and target types.
// It generates a standard relationship type based on the memory types of the source and target.
//
// Parameters:
//   - sourceType: The memory type of the source node
//   - targetType: The memory type of the target node
//
// Returns:
//   - A string representing the inferred relationship type
func InferRelationshipType(sourceType, targetType string) string {
	return fmt.Sprintf("HAS_%s", strings.ToUpper(targetType))
}

// buildCypherQuery builds a Cypher query string from the templates and parameters.
// It constructs a Cypher query by replacing placeholders in the template with actual values.
//
// Parameters:
//   - template: The Cypher query template with placeholders
//   - params: A map of parameters to replace in the template
//
// Returns:
//   - The constructed Cypher query string
func buildCypherQuery(template string, params map[string]interface{}) string {
	query := template
	for key, value := range params {
		placeholder := fmt.Sprintf("${%s}", key)
		var replacement string

		switch v := value.(type) {
		case string:
			replacement = fmt.Sprintf("'%s'", v)
		case int, int64, float64:
			replacement = fmt.Sprintf("%v", v)
		case bool:
			replacement = fmt.Sprintf("%t", v)
		case []string:
			var quotedValues []string
			for _, s := range v {
				quotedValues = append(quotedValues, fmt.Sprintf("'%s'", s))
			}
			replacement = fmt.Sprintf("[%s]", strings.Join(quotedValues, ", "))
		default:
			// For complex types, use JSON representation
			jsonStr, _ := json.Marshal(v)
			replacement = string(jsonStr)
		}

		query = strings.Replace(query, placeholder, replacement, -1)
	}

	return query
}

// sanitizeCypherString sanitizes a string for use in a Cypher query.
// It escapes single quotes in the input string to prevent Cypher injection.
//
// Parameters:
//   - input: The string to sanitize
//
// Returns:
//   - The sanitized string
func sanitizeCypherString(input string) string {
	return strings.Replace(input, "'", "\\'", -1)
}
