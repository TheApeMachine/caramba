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
	"github.com/theapemachine/errnie"
)

/*
Researcher is an example agent to demonstrate tool usage, memory, planning, and optimization.
*/
type Researcher struct {
	Agent   core.Agent
	Planner core.Planner
}

/*
NewResearcher demonstrates how to build up an agent with tools, memory, planner, and optimizer.
*/
func NewResearcher() *Researcher {
	// Print descriptive title
	output.Title("CARAMBA RESEARCH AGENT")
	output.Info("Initializing research agent with browser capabilities and guided planning")

	return &Researcher{
		Agent: core.NewAgentBuilder(
			"ResearchAgent",
		).WithLLM(
			llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
		).WithSystemPrompt(
			viper.GetViper().GetString("templates.researcher"),
		).WithPlanner(
			core.NewAgentBuilder(
				"PlannerAgent",
			).WithLLM(
				llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
			).WithSystemPrompt(
				viper.GetViper().GetString("templates.planner"),
			).Build(),
		).WithOptimizer(
			core.NewAgentBuilder(
				"OptimizerAgent",
			).WithLLM(
				llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
			).WithSystemPrompt(
				viper.GetViper().GetString("templates.optimizer"),
			).Build(),
		).WithTool(
			tools.NewBrowserTool("http://localhost:3000", "6R0W53R135510"),
		).WithMemory(
			memory.NewUnifiedMemory(memory.NewInMemoryStore(), memory.DefaultUnifiedMemoryOptions()),
		).WithIterationLimit(
			10,
		).Build(),
	}
}

func (researcher *Researcher) Run(ctx context.Context, query string) (string, error) {
	errnie.Info("Starting researcher example")

	// Print what we're researching
	output.Title("RESEARCHING: " + query)

	return researcher.Agent.Execute(ctx, core.LLMMessage{
		Role:    "user",
		Content: query,
	})
}
