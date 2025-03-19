package examples

import (
	"bufio"
	"fmt"
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Chat struct {
	agent    *ai.Agent
	provider *provider.OpenAIProvider
	workflow io.ReadWriter
}

func NewChat() *Chat {
	errnie.Debug("examples.NewChat")

	agent := ai.NewAgent()
	provider := provider.NewOpenAIProvider(
		provider.WithAPIKey(core.NewConfig().OpenAIAPIKey),
	)

	chat := &Chat{
		agent:    agent,
		provider: provider,
		workflow: workflow.NewPipeline(
			agent,
			workflow.NewFeedback(
				provider,
				agent,
			),
			workflow.NewConverter(),
			os.Stdout,
		),
	}

	return chat
}

func (chat *Chat) Run() (err error) {
	errnie.Info("Starting pipeline example")

	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("> ")
		input, err := reader.ReadString('\n')

		if err != nil {
			return err
		}

		msg := datura.New(
			datura.WithPayload(provider.NewParams(
				provider.WithMessages(
					provider.NewMessage(
						provider.WithUserRole("Danny", input),
					),
				),
				provider.WithModel(tweaker.GetModel(tweaker.GetProvider())),
				provider.WithTemperature(tweaker.GetTemperature()),
				provider.WithStream(tweaker.GetStream()),
			).Marshal()),
		)

		if _, err = io.Copy(chat, msg); err != nil && err != io.EOF {
			return err
		}
	}

	return nil
}

func (chat *Chat) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Chat.Read")
	return chat.workflow.Read(p)
}

func (chat *Chat) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Chat.Write")
	return chat.workflow.Write(p)
}
