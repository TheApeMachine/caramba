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

func RunBrowserAgentExample() {
	v := viper.GetViper()
	system := v.GetString("prompts.templates.systems.default")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if system == "" {
		errnie.Error(errors.New("system is empty"))
		return
	}

	prvdr := provider.NewBalancedProvider()
	err := prvdr.Initialize(ctx)
	if err != nil {
		errnie.Error(err)
		return
	}

	dctx := drknow.QuickContext(system)
	agent := ai.NewAgent(
		dctx,
		prvdr,
		"researcher",
		10,
	)

	stream.NewConsumer().Print(
		agent.Generate(
			ctx,
			provider.NewMessage(
				provider.RoleUser,
				`Please demonstrate the browser tool's capabilities by: 
				  1. Getting the top 3 trending Go projects from GitHub using the pre-made extractor 
				  2. Extracting the main article content from a tech blog using the common article extractor 
				  3. Building a custom extractor to get specific information from a webpage`,
			),
		),
		false,
	)
}
