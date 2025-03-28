package tools

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/system"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type SystemTool struct {
	buffer   *stream.Buffer
	Schema   *provider.Tool
	registry *system.Registry
}

func NewSystemTool() *SystemTool {
	registry := system.NewRegistry()

	return &SystemTool{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("system.NewSystemTool.buffer.fn")

			if err = workflow.NewFlipFlop(artifact, registry); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		Schema:   GetToolSchema("system"),
		registry: registry,
	}
}
