package tools

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/environment"
)

/*
Environment is a tool that allows the AI agent to interact with a full Linux terminal-based
environment. It uses Docker containers to create a somewhat isolated environment for the agent
to run in.

In the future, this should be enhanced with a GVisor layer to further sandbox the environment.
*/
type Environment struct {
	buffer *stream.Buffer
	runner *environment.Runner
}

/*
NewEnvironment creates a new Environment tool.
*/
func NewEnvironment() *Environment {
	errnie.Debug("environment.NewEnvironment")

	var runner *environment.Runner

	environment := &Environment{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) error {
			errnie.Debug("environment.Instance.buffer.fn")

			builder := environment.NewBuilder()
			errnie.Error(builder.Container.Load())
			runner = environment.NewRunner(builder.Container)

			return nil
		}),
		runner: runner,
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
