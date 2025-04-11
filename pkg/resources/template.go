package resources

import (
	"fmt"
	"net/url"
	"strings"
)

// TemplateVariable represents a variable in a URI template
type TemplateVariable struct {
	Name        string
	Description string
	Required    bool
}

// ParseTemplateVariables extracts variables from a URI template
func ParseTemplateVariables(template string) []TemplateVariable {
	variables := []TemplateVariable{}

	// Find all variables in the template (e.g., {category}, {name})
	parts := strings.Split(template, "{")
	for i := 1; i < len(parts); i++ {
		part := parts[i]
		endIndex := strings.Index(part, "}")
		if endIndex == -1 {
			continue
		}

		varName := part[:endIndex]
		variables = append(variables, TemplateVariable{
			Name:     varName,
			Required: true, // All variables are required by default
		})
	}

	return variables
}

// ExpandTemplate expands a URI template with provided variables
func ExpandTemplate(template string, variables map[string]string) (string, error) {
	result := template

	// Extract variables from the template
	templateVars := ParseTemplateVariables(template)

	// Check if all required variables are provided
	for _, v := range templateVars {
		if v.Required {
			if _, ok := variables[v.Name]; !ok {
				return "", fmt.Errorf("missing required variable: %s", v.Name)
			}
		}
	}

	// Replace variables in the template
	for varName, varValue := range variables {
		placeholder := "{" + varName + "}"
		result = strings.ReplaceAll(result, placeholder, varValue)
	}

	// Validate the expanded URI
	if _, err := url.Parse(result); err != nil {
		return "", fmt.Errorf("invalid URI after expansion: %w", err)
	}

	return result, nil
}

// ValidateTemplate validates a URI template
func ValidateTemplate(template string) error {
	// Check if the template contains at least one variable
	if !strings.Contains(template, "{") || !strings.Contains(template, "}") {
		return fmt.Errorf("template must contain at least one variable: %s", template)
	}

	// Check if all variables are properly closed
	openCount := strings.Count(template, "{")
	closeCount := strings.Count(template, "}")
	if openCount != closeCount {
		return fmt.Errorf("mismatched braces in template: %s", template)
	}

	// Try to parse the template to ensure it's a valid URI
	// Replace variables with placeholder values for validation
	testTemplate := template
	for i := 0; i < openCount; i++ {
		testTemplate = strings.Replace(testTemplate, "{", "test", 1)
		testTemplate = strings.Replace(testTemplate, "}", "", 1)
	}

	if _, err := url.Parse(testTemplate); err != nil {
		return fmt.Errorf("template is not a valid URI: %w", err)
	}

	return nil
}
