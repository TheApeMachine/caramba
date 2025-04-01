package tools

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
)

func init() {
	provider.RegisterTool("system")
}

type SystemTool struct {
	Schema    *provider.Tool
	operation string
}

func NewSystemTool(operation string) *SystemTool {
	return &SystemTool{
		Schema:    GetToolSchema("system"),
		operation: operation,
	}
}

func (tool *SystemTool) Generate(buffer chan *datura.Artifact) chan *datura.Artifact {
	errnie.Debug("system.SystemTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		artifact := <-buffer

		switch tool.operation {
		case "inspect":
			artifact.SetMetaValue("operation", "inspect")
		case "optimize":
			artifact.SetMetaValue("operation", "optimize")
		case "message":
			artifact.SetMetaValue("operation", "message")
		}
	}()

	return out
}
