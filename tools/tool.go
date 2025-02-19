package tools

import (
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/provider"
)

// Tool defines the interface that all tools must implement
type Tool interface {
	Convert() provider.Tool
	Use(agent AgentTool, artifact *datura.Artifact)
}
