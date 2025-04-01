package tools

type Toolset struct {
	tools []*Tool
}

type ToolsetOption func(*Toolset)

func NewToolset(opts ...ToolsetOption) *Toolset {
	toolset := &Toolset{
		tools: []*Tool{},
	}

	for _, opt := range opts {
		opt(toolset)
	}

	return toolset
}

func WithTools(tools ...*Tool) ToolsetOption {
	return func(toolset *Toolset) {
		toolset.tools = append(toolset.tools, tools...)
	}
}
