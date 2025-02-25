package tools

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"text/template"
)

// Formatter is a tool for formatting strings using templates
type Formatter struct{}

// NewFormatter creates a new Formatter tool
func NewFormatter() *Formatter {
	return &Formatter{}
}

// Name returns the name of the tool
func (f *Formatter) Name() string {
	return "formatter"
}

// Description returns the description of the tool
func (f *Formatter) Description() string {
	return "A tool for formatting strings using Go templates"
}

// Execute executes the formatter with the given arguments
func (f *Formatter) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	templateStr, ok := args["template"].(string)
	if !ok {
		return nil, errors.New("template argument must be a string")
	}

	// Create a template data context with all args
	// This allows accessing values from previous steps
	data := make(map[string]interface{})
	for k, v := range args {
		data[k] = v
	}

	// Create a new template
	tmpl, err := template.New("formatter").Parse(templateStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}

	// Execute the template with the provided data
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return nil, fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

// Schema returns the JSON schema for the tool's arguments
func (f *Formatter) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"template": map[string]interface{}{
				"type":        "string",
				"description": "The template string to format",
			},
		},
		"required": []string{"template"},
	}
}
