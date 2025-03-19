package examples

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
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
		ai.WithTools([]*provider.Tool{
			tools.NewEditorTool().Schema,
			tools.NewEnvironment().Schema,
		}),
	)

	return &Code{
		agent: agent,
		provider: provider.NewOpenAIProvider(
			os.Getenv("OPENAI_API_KEY"),
			tweaker.GetEndpoint("openai"),
		),
		workflow: workflow.NewPipeline(
			agent,
		),
	}
}
