package examples

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Chat struct {
	workflow io.ReadWriter
}

func NewChat() *Chat {
	errnie.Debug("examples.NewChat")

	return &Chat{
		workflow: workflow.NewPipeline(
			workflow.NewFeedback(
				provider.NewOpenAIProvider(
					provider.WithAPIKey(core.NewConfig().OpenAIAPIKey),
				),
				ai.NewAgent(),
			),
			workflow.NewConverter(),
			os.Stdout,
		),
	}
}

func (chat *Chat) Run() (err error) {
	errnie.Info("Starting pipeline example")

	var (
		input     string
		shouldExit = false
	)

	for !shouldExit {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("> ")
		
		if input, err = reader.ReadString('\n'); err != nil {
			return err
		}

		if strings.HasPrefix(input, "quit") {
			shouldExit = true
		}

		if _, err = io.Copy(chat, datura.New(
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
		)); err != nil && err != io.EOF {
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
