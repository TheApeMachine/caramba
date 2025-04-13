package agents

import "github.com/theapemachine/caramba/pkg/registry"

func init() {
	reg := registry.GetAmbient()
	reg.RegisterTool("list_agents", NewList())
	reg.RegisterTool("agent_card", NewCard())
}
