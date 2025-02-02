package examples

import (
	"context"
	"errors"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/errnie"
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
	))
}
