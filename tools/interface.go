package tools

import (
	"context"
	"io"
	"strings"
)

type Tool interface {
	Name() string
	Description() string
	GenerateSchema() interface{}
	Use(map[string]any) string
	Connect(context.Context, io.ReadWriteCloser) error
}

var defaultTools = map[string]Tool{
	"show":  &Show{},
	"send":  &Send{},
	"break": &Break{},
}

type Toolset struct {
	tools map[string]Tool
}

func NewToolset(tools ...Tool) *Toolset {
	set := defaultTools

	for _, tool := range tools {
		set[tool.Name()] = tool
	}

	return &Toolset{
		tools: set,
	}
}

func (toolset *Toolset) Use(
	toolName string,
	input map[string]any,
) string {
	if tool, ok := toolset.tools[toolName]; ok {
		return tool.Use(input)
	}
	return ""
}

func (toolset *Toolset) String() string {
	schemas := []string{}

	for _, tool := range toolset.tools {
		schemas = append(schemas, tool.GenerateSchema().(string))
	}

	return strings.Join(schemas, "\n\n")
}
