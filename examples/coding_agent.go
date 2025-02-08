package examples

import (
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
)

type CodingAgent struct{}

func NewCodingAgent() *CodingAgent {
	return &CodingAgent{}
}

func (c *CodingAgent) Run() {
	pool := ai.NewAgentPool()

	pool.AddAgent(ai.NewAgent(
		drknow.QuickContext("You are a developer"),
		provider.NewBalancedProvider(),
		"developer",
		10,
	))
}
