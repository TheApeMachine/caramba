package tools

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/environment"
	"github.com/theapemachine/caramba/pkg/workflow"
)

func init() {
	provider.RegisterTool("environment")
}

/*
Environment is a tool that allows the AI agent to interact with a full Linux terminal-based
environment. It uses Docker containers to create a somewhat isolated environment for the agent
to run in.

In the future, this should be enhanced with a GVisor layer to further sandbox the environment.
*/
type Environment struct {
	buffer  *stream.Buffer
	builder *environment.Builder
	runner  *environment.Runner
	Schema  *provider.Tool
}

/*
NewEnvironment creates a new Environment tool.
*/
func NewEnvironment() *Environment {
	errnie.Debug("environment.NewEnvironment")

	// Setup the docker container for the agent's environment tool.
	builder := environment.NewBuilder()

	if errnie.Error(builder.Container.Load()) != nil {
		return nil
	}

	runner := environment.NewRunner(builder.Container)

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
		Schema: provider.NewTool(
			provider.WithFunction(
				"environment",
				"A tool which gives you a full Linux terminal-based environment to interact with.",
			),
			provider.WithProperty(
				"command",
				"string",
				"A valid bash command to execute in the environment. Note: Be careful not to execute any infinitely blocking commands, you will only be able to continue once the last executed command has finished.",
				[]any{},
			),
		),
	}

	return environment
}

/*
Read reads data from the environment.
*/
func (environment *Environment) Read(p []byte) (n int, err error) {
	errnie.Debug("environment.Environment.Read")
	return environment.buffer.Read(p)
}

/*
Write writes data to the environment.
*/
func (environment *Environment) Write(p []byte) (n int, err error) {
	errnie.Debug("environment.Environment.Write")
	return environment.buffer.Write(p)
}

/*
Close closes the environment.
*/
func (environment *Environment) Close() error {
	errnie.Debug("environment.Environment.Close")
	return environment.buffer.Close()
}
