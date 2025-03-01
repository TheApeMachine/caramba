package examples

import (
	"context"
	"os"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/caramba/pkg/process"
)

/*
Researcher is an example agent to demonstrate tool usage, memory, planning, and optimization.
*/
type Researcher struct {
	logger  *output.Logger
	Agent   core.Agent
	Planner core.Planner
}

/*
NewResearcher demonstrates how to build up an agent with tools, memory, planner, and optimizer.
*/
func NewResearcher() *Researcher {
	builder := func(name string) core.Agent {
		return core.NewAgentBuilder(
			name,
		).WithLLM(
			llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
		).WithSystemPrompt(
			viper.GetViper().GetString("templates." + name),
		).WithProcess(
			&process.Plan{},
		).WithStreaming(
			true,
		).Build()
	}

	planner := builder("planner")
	optimizer := builder("optimizer")

	return &Researcher{
		logger: output.NewLogger(),
		Agent: core.NewAgentBuilder(
			"researcher",
		).WithLLM(
			llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
		).WithSystemPrompt(
			viper.GetViper().GetString("templates.researcher"),
		).WithPlanner(
			planner,
		).WithOptimizer(
			optimizer,
		).WithTool(
			tools.NewBrowserTool("http://localhost:3000", "6R0W53R135510"),
		).WithMemory(
			memory.NewUnifiedMemory(),
		).WithIterationLimit(
			10,
		).WithStreaming(
			true,
		).Build(),
	}
}

func (researcher *Researcher) Run(ctx context.Context) (string, error) {
	researcher.logger.Log("Running researcher")
	return researcher.Agent.Execute(ctx)
}
