package examples

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
Code demonstrates an example workflow that uses an AI agent to generate
and execute code. It showcases the integration between AI providers,
workflow pipelines, and feedback mechanisms.
*/
type Code struct {
	ctx     context.Context
	cancel  context.CancelFunc
	streams []*core.Streamer
}

/*
NewCode creates a new Code example instance with a configured AI agent,
OpenAI provider, and workflow pipeline. It sets up all necessary components
for code generation and execution.

Returns:
  - *Code: A new Code instance ready to run the example
*/
func NewCode() *Code {
	errnie.Debug("examples.NewCode")

	ctx, cancel := context.WithCancel(context.Background())
	streams := make([]*core.Streamer, 0)

	streams = append(streams, core.NewStreamer(
		ai.NewAgentBuilder(
			ai.WithCancel(ctx),
			ai.WithIdentity(utils.GenerateName(), "teamlead"),
			ai.WithProvider(provider.ProviderTypeOpenAI),
			ai.WithParams(ai.NewParamsBuilder(
				ai.WithModel("gpt-4o-mini"),
				ai.WithTemperature(0.5),
			)),
			ai.WithContext(
				ai.NewContextBuilder(
					ai.WithMessages(
						ai.NewMessageBuilder(
							ai.WithRole("user"),
							ai.WithContent("Write a Python game"),
						),
					),
				),
			),
			ai.WithTools(
				tools.NewToolBuilder(tools.WithMCP(tools.NewMemoryTool().Schema.ToMCP())),
				tools.NewToolBuilder(tools.WithMCP(tools.NewAzure().Schema.ToMCP())),
				tools.NewToolBuilder(tools.WithMCP(tools.NewSystemInspectTool().Schema.ToMCP())),
				tools.NewToolBuilder(tools.WithMCP(tools.NewSystemOptimizeTool().Schema.ToMCP())),
				tools.NewToolBuilder(tools.WithMCP(tools.NewSystemMessageTool().Schema.ToMCP())),
			),
		),
	))

	// Note that this time we have os.Stdout as the last argument.
	// This is because we want to output the code to the console.
	// And the Pump we will use later will never return.
	return &Code{
		ctx:     ctx,
		cancel:  cancel,
		streams: streams,
	}
}

/*
Generate executes the code example workflow. It sends an initial message to the AI
requesting a Python game implementation, processes the response through the
workflow pipeline, and outputs the results.

Returns:
  - error: Any error that occurred during execution
*/
func (code *Code) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Info("Starting code example")

	out := make(chan *datura.Artifact)

	go func() {
		for {
			select {
			case <-code.ctx.Done():
				errnie.Info("Code example cancelled")
				code.cancel()
				return
			case artifact := <-buffer:
				for _, stream := range code.streams {
					if _, err := io.Copy(stream, artifact); errnie.Error(err) != nil {
						continue
					}
				}
			}
		}
	}()

	return out
}
