package tools

import "sync"

var once sync.Once
var registry *Registry

type Registry struct {
	Tools []Tool
}

func NewRegistry() *Registry {
	once.Do(func() {
		registry = &Registry{}
	})

	return registry
}

func (registry *Registry) Register(tool Tool) {
	registry.Tools = append(registry.Tools, tool)
}

func (registry *Registry) GetToolNames() []string {
	names := make([]string, len(registry.Tools))

	for i, tool := range registry.Tools {
		names[i] = tool.Tool.Name
	}

	return names
}
