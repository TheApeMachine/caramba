package examples

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

/*
Pipeline demonstrates a basic pipeline, tool calling, and a feedback loop.
*/
type Pipeline struct {
	agent    *ai.Agent
	provider *provider.OpenAIProvider
	workflow io.ReadWriter
}

/*
NewPipeline sets up the Pipeline example.
*/
func NewPipeline() *Pipeline {
	errnie.Debug("examples.NewPipeline")

	// You need a caller for the agent to call tools.
	caller := tools.NewCaller()

	// Create the agent with the caller.
	agent := ai.NewAgent(ai.WithCaller(caller))

	// Set up the provider with the API key.
	provider := provider.NewOpenAIProvider(
		provider.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)

	// Create a converter, which will extract the Content field from the current
	// Artifact and stream it out using a bytes.Buffer, keeping up with the
	// "everything is io" paradigm.
	converter := workflow.NewConverter()

	// Create the workflow, which is a pipeline in this case. A workflow is just
	// any type of way you can connect io.ReadWriter components together.
	// Interesting to note is the Feedback component, which is just an io.TeeReader
	// that sends the output of the provider both forwards to the converter, and
	// backwards to the agent as a new message. This allows us to both have a real-time
	// output stream, and a way for the agent to retain context (conversation history).
	pipeline := &Pipeline{
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

	return pipeline
}

func (pipeline *Pipeline) Run() (err error) {
	errnie.Info("Starting pipeline example")

	// One challenge with "everything is io" is the predictability of the messages.
	// Having different types of marshaled messages makes it practically impossible
	// to connect components in an arbitrary way. datura.Artifact is a type that
	// solves for this by being a wrapper that describes the content it wraps.
	// Building this on top of Cap'n Proto provides the most cost-effective way
	// regarding compute and memory while converting between its wire format and
	// the Artifact type.
	msg := datura.New(
		datura.WithPayload(provider.NewParams(
			provider.WithModel("gpt-4o-mini"),
			provider.WithTopP(1),
			provider.WithTools(
				tools.NewBrowser().Schema,
			),
			provider.WithMessages(
				provider.NewMessage(
					provider.WithSystemRole(
						tweaker.GetSystemPrompt("default"),
					),
				),
				provider.NewMessage(
					provider.WithUserRole(
						"Danny",
						"Research the state of the art in AI agents, then formulate a plan for a new AI agent that can help me achieve my goals.",
					),
				),
			),
		).Marshal()),
	)

	// Remember, "everything is io", so once you're all set up, you can just
	// copy from one component to another.
	if _, err = io.Copy(pipeline, msg); err != nil && err != io.EOF {
		return err
	}

	// Read from the end of the pipeline, and send it to an output, in this
	// case, stdout, to show the output on the console.
	if _, err = io.Copy(os.Stdout, pipeline); err != nil && err != io.EOF {
		return err
	}

	return nil
}

/*
Read implements io.Reader for the Pipeline example. In almost all cases,
you will not have to do much more than proxy a component's Read method,
most often (for internal components) that would be through a stream.Buffer.
Here we use the workflow's Read method, which is using a Pipeline.
*/
func (pipeline *Pipeline) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Read")
	return pipeline.workflow.Read(p)
}

/*
Write implements io.Writer for the Pipeline example. In almost all cases,
you will not have to do much more than proxy a component's Write method,
most often (for internal components) that would be through a stream.Buffer.
Here we use the workflow's Write method, which is using a Pipeline.
*/
func (pipeline *Pipeline) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Pipeline.Write")
	return pipeline.workflow.Write(p)
}
