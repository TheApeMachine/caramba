package tools

import (
	"context"
	"io"
	"strings"

	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/types"
)

type Tool interface {
	Name() string
	Description() string
	GenerateSchema() any
	Use(*stream.Accumulator, map[string]any, ...types.Generator) *stream.Accumulator
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
	accumulator *stream.Accumulator,
	toolName string,
	input map[string]any,
	generators ...types.Generator,
) *stream.Accumulator {
	if tool, ok := toolset.tools[toolName]; ok {
		return tool.Use(accumulator, input, generators...)
	}
	return nil
}

func (toolset *Toolset) String() string {
	schemas := []string{}

	for _, tool := range toolset.tools {
		schemas = append(schemas, tool.GenerateSchema().(string))
	}

	return strings.Join(schemas, "\n\n")
}
