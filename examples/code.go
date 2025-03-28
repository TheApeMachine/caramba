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
Code demonstrates an example workflow that uses an AI agent to generate
and execute code. It showcases the integration between AI providers,
workflow pipelines, and feedback mechanisms.
*/
type Code struct {
	agent    *ai.Agent
	provider *provider.OpenAIProvider
	workflow io.ReadWriter
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

	agent := ai.NewAgent(ai.WithCaller(tools.NewCaller()))

	provider := provider.NewOpenAIProvider()

	converter := workflow.NewConverter()

	// Note that this time we have os.Stdout as the last argument.
	// This is because we want to output the code to the console.
	// And the Pump we will use later will never return.
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

/*
Run executes the code example workflow. It sends an initial message to the AI
requesting a Python game implementation, processes the response through the
workflow pipeline, and outputs the results.

Returns:
  - error: Any error that occurred during execution
*/
func (code *Code) Run() (err error) {
	errnie.Info("Starting code example")

	msg := datura.New(
		datura.WithPayload(provider.NewParams(
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
		).Marshal()),
	)

	errnie.Info("copying msg to code")
	if _, err = io.Copy(code, msg); err != nil && err != io.EOF {
		return err
	}

	for range 3 {
		errnie.Info("copying code to stdout")
		if _, err = io.Copy(os.Stdout, code); err != nil && err != io.EOF {
			return err
		}
	}

	return nil
}

/*
Read implements the io.Reader interface for the Code example.
It delegates reading operations to the underlying workflow.

Parameters:
  - p: Byte slice to read data into

Returns:
  - n: Number of bytes read
  - err: Any error that occurred during reading
*/
func (code *Code) Read(p []byte) (n int, err error) {
	errnie.Debug("examples.Code.Read")
	return code.workflow.Read(p)
}

/*
Write implements the io.Writer interface for the Code example.
It delegates writing operations to the underlying workflow.

Parameters:
  - p: Byte slice containing data to write

Returns:
  - n: Number of bytes written
  - err: Any error that occurred during writing
*/
func (code *Code) Write(p []byte) (n int, err error) {
	errnie.Debug("examples.Code.Write")
	return code.workflow.Write(p)
}

/*
Close implements the io.Closer interface for the Code example.
It signals shutdown via the done channel and closes the underlying workflow.

Returns:
  - error: Any error that occurred during closure
*/
func (code *Code) Close() (err error) {
	errnie.Debug("examples.Code.Close")
	return nil
}
