package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/tweaker"
)

type Config struct {
	Name         string
	Role         string
	SystemPrompt string
	Thread       *provider.Thread
	Toolset      *tools.Toolset
	Temperature  float32
}

func NewConfig(system, role, name string, toolset *tools.Toolset) *Config {
	return &Config{
		Name:         name,
		Role:         role,
		SystemPrompt: system,
		Thread: provider.NewThread(
			provider.NewMessage(provider.RoleSystem,
				tweaker.GetSystemPrompt(
					system, name, role, toolset.String(),
				),
			),
		),
		Toolset:     toolset,
		Temperature: 0.1,
	}
}
