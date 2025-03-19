package examples

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Code struct {
	agent    *ai.Agent
	provider *provider.OpenAIProvider
	workflow io.ReadWriter
}

func NewCode() *Code {
	errnie.Debug("examples.NewCode")

	agent := ai.NewAgent(
		ai.WithModel("gpt-4o-mini"),
		ai.WithTools(
			tools.NewEditorTool().Schema,
			tools.NewEnvironment().Schema,
		),
	)

	provider := provider.NewOpenAIProvider(
		provider.WithAPIKey(core.NewConfig().OpenAIAPIKey),
	)

	converter := workflow.NewConverter()

	return &Code{
		agent:    agent,
		provider: provider,
		workflow: workflow.NewPipeline(
			agent,
			workflow.NewFeedback(
				provider,
				agent,
			),
			converter,
		),
	}
}

func (code *Code) Run() (err error) {
	errnie.Info("Starting code example")

	msg := datura.New(
		datura.WithPayload(provider.NewParams(
			provider.WithModel("gpt-4o-mini"),
			provider.WithTools(
				tools.NewEditorTool().Schema,
				tools.NewEnvironment().Schema,
			),
			provider.WithTopP(1),
			provider.WithMessages(
				provider.NewMessage(
					provider.WithSystemRole(
						tweaker.GetSystemPrompt("code"),
					),
				),
				provider.NewMessage(
					provider.WithUserRole(
						"Danny",
						"Please write a simple game using Python",
					),
				),
			),
		).Marshal()),
	)

	if _, err = io.Copy(code, msg); err != nil && err != io.EOF {
		return err
	}

	if _, err = io.Copy(os.Stdout, code); err != nil && err != io.EOF {
		return err
	}

	return nil
}

func (code *Code) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Code.Read")
	return code.workflow.Read(p)
}

func (code *Code) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Code.Write")
	return code.workflow.Write(p)
}
