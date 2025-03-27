package examples

import (
	"io"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

var providers = []io.ReadWriteCloser{
	provider.NewOpenAIProvider(),
	provider.NewAnthropicProvider(),
	provider.NewGoogleProvider(),
}

var ctxs = []*provider.Params{
	provider.NewParams(
		provider.WithModel("gpt-4o-mini"),
		provider.WithTools(
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
					"Please write a simple game using Python. You have to run it, and play a round, to verify it works, so use an environment.",
				),
			),
		),
	),
}

type Discussion struct {
	participants []io.ReadWriter
	workflow     io.ReadWriter
}

func NewDiscussion() *Discussion {
	discussion := &Discussion{}

	for idx := range 3 {
		discussion.participants = append(discussion.participants, ai.NewAgent(
			ai.WithCaller(tools.NewCaller()),
		))

		discussion.participants = append(discussion.participants, providers[idx])
	}

	return &Discussion{
		workflow: workflow.NewPipeline(discussion.participants...),
	}
}

func (discussion *Discussion) Run() (err error) {
	return nil
}
