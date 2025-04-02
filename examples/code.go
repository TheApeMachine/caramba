package examples

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/system"
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
			ai.WithParams(core.NewParamsBuilder(
				core.WithModel("gpt-4o-mini"),
				core.WithTemperature(0.5),
			)),
			ai.WithContext(
				core.NewContextBuilder(
					core.WithMessages(
						core.NewMessageBuilder(
							core.WithRole("user"),
							core.WithContent("Write a Python game"),
						),
					),
				),
			),
			ai.WithTools("memory", "system_inspect", "system_optimize", "system_message"),
		),
	))

	streams = append(streams, core.NewStreamer(
		provider.NewProviderBuilder(
			provider.WithCancel(ctx),
			provider.WithSupplier("openai"),
		),
	))

	streams = append(streams, core.NewStreamer(tools.NewMemoryTool()))
	streams = append(streams, core.NewStreamer(tools.NewSystemInspectTool()))
	streams = append(streams, core.NewStreamer(tools.NewSystemOptimizeTool()))
	streams = append(streams, core.NewStreamer(tools.NewSystemMessageTool()))

	// Note that this time we have os.Stdout as the last argument.
	// This is because we want to output the code to the console.
	// And the Pump we will use later will never return.
	return &Code{
		ctx:     ctx,
		cancel:  cancel,
		streams: streams,
	}
}

func (code *Code) ID() string {
	return "code_example"
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
	hub := system.NewHub()

	// Register each stream component with the hub
	for i, stream := range code.streams {
		clientID := fmt.Sprintf("stream-%d", i)
		system.WithClient(clientID, stream)(hub)

		// Subscribe to topics based on component type
		switch i {
		case 0: // Agent
			system.WithTopics(clientID, "agent", "tools")(hub)
		case 1: // Provider
			system.WithTopics(clientID, "provider", "agent")(hub)
		default: // Tools
			system.WithTopics(clientID, "tools")(hub)
		}
	}

	// Start the hub processing with the input buffer
	hubChannel := hub.Generate(buffer)

	go func() {
		defer close(out)

		for {
			select {
			case <-code.ctx.Done():
				errnie.Info("Code example cancelled")
				code.cancel()
				return
			case artifact, ok := <-buffer:
				if !ok {
					return
				}
				// Add routing metadata to the artifact
				datura.WithMeta("topic", system.Topic("broadcast"))(artifact)

				// Echo the artifact to the output channel
				out <- artifact
			case artifact, ok := <-hubChannel:
				if !ok {
					return
				}
				// Forward hub processed artifacts to output
				out <- artifact
			}
		}
	}()

	return out
}
