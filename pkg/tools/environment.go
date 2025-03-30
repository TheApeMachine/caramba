package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/environment"
	"github.com/theapemachine/caramba/pkg/workflow"
)

/*
init registers the environment tool with the provider system.
*/
func init() {
	fmt.Println("tools.environment.init")
	provider.RegisterTool("environment")
}

/*
Environment provides a sandboxed Linux environment for executing commands.

It uses Docker containers to create an isolated environment for running commands
and interacting with a full Linux terminal. The environment is managed through
a builder for setup and a runner for command execution.

Future enhancements will include GVisor for additional sandboxing capabilities.
*/
type Environment struct {
	buffer  *stream.Buffer
	builder *environment.Builder
	runner  *environment.Runner
	Schema  *provider.Tool
}

/*
NewEnvironment creates a new Environment tool instance.

It initializes a Docker container environment with necessary runtime components
and sets up a buffered stream for processing commands. Returns nil if either
the builder or runner initialization fails.
*/
func NewEnvironment() *Environment {
	errnie.Debug("environment.NewEnvironment")

	// Setup the docker container for the agent's environment tool.
	builder := environment.NewBuilder()
	if builder == nil {
		return nil
	}

	runner := environment.NewRunner(builder.Runtime)
	if runner == nil {
		return nil
	}

	environment := &Environment{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("environment.Instance.buffer.fn")

			// After the FlipFlop the output of the command is in the payload of the artifact.
			if err = workflow.NewFlipFlop(artifact, runner); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		builder: builder,
		runner:  runner,
		Schema:  GetToolSchema("environment"),
	}

	return environment
}

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after environment operations
have been completed.
*/
func (environment *Environment) Read(p []byte) (n int, err error) {
	errnie.Debug("environment.Environment.Read")
	return environment.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing by
the environment runner.
*/
func (environment *Environment) Write(p []byte) (n int, err error) {
	errnie.Debug("environment.Environment.Write")
	return environment.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (environment *Environment) Close() error {
	errnie.Debug("environment.Environment.Close")
	return environment.buffer.Close()
}
