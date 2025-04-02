package tools

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/environment"
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
		builder: builder,
		runner:  runner,
		Schema:  GetToolSchema("environment"),
	}

	return environment
}

func (env *Environment) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return env.runner.Generate(buffer)
}
