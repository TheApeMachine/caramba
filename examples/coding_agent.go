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

type CodingAgent struct {
	ctx    context.Context
	cancel context.CancelFunc
	agent  *ai.Agent
}

func NewCodingAgent() *CodingAgent {
	v := viper.GetViper()
	system := v.GetString("prompts.templates.systems.default")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if system == "" {
		errnie.Error(errors.New("system is empty"))
		return nil
	}

	prvdr := provider.NewBalancedProvider()
	err := prvdr.Initialize(ctx)
	if err != nil {
		errnie.Error(err)
		return nil
	}

	dctx := drknow.QuickContext(system)
	agent := ai.NewAgent(
		dctx,
		prvdr,
		"researcher",
		10,
	)

	return &CodingAgent{
		ctx:    ctx,
		cancel: cancel,
		agent:  agent,
	}
}

func (a *CodingAgent) Run() {
	defer a.cancel()

	stream.NewConsumer().Print(
		a.agent.Generate(
			a.ctx,
			provider.NewMessage(
				provider.RoleUser,
				`Please write a simple Go program that prints "Hello, World!" to the console.`,
			),
		),
		false,
	)
}
