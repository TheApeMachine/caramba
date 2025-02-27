package examples

import (
	"context"
	"fmt"
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
Researcher is an example agent that uses the browser tool with guided planning.
*/
type Researcher struct {
	Agent   core.Agent
	Planner core.Planner
}

func NewResearcher() *Researcher {
	// Print descriptive title
	output.Title("CARAMBA RESEARCH AGENT")
	output.Info("Initializing research agent with browser capabilities and guided planning")

	// Create browser tool with additional logging
	browserTool := tools.NewBrowserTool("http://localhost:3000", "6R0W53R135510")

	// Create LLM provider to be used by both agent and planner
	llmProvider := llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini")

	// Create a tools map for the planner and agent
	toolsMap := map[string]core.Tool{
		"browser": browserTool,
	}

	// Create the agent
	agent := core.NewAgentBuilder(
		"BrowserAgent",
	).WithLLM(
		llmProvider,
	).WithSystemPrompt(viper.GetViper().GetString("templates.researcher")).WithTool(
		browserTool,
	).WithMemory(
		memory.NewUnifiedMemory(memory.NewInMemoryStore(), memory.DefaultUnifiedMemoryOptions()),
	).WithIterationLimit(
		10,
	).Build()

	// Create the planner
	planner := core.NewSimplePlanner(llmProvider, toolsMap)

	output.Result("Research agent successfully initialized")

	return &Researcher{
		Agent:   agent,
		Planner: planner,
	}
}

func (researcher *Researcher) Run(ctx context.Context, query string) (string, error) {
	errnie.Info("Starting researcher example")

	// Print what we're researching
	output.Title("RESEARCHING: " + query)

	// Stage 1: Planning approach
	output.Stage(1, "Planning research approach")
	planningSpinner := output.StartSpinner("Agent is analyzing the query and planning research strategy")

	// Create a research plan using the planner
	plan, err := researcher.Planner.CreatePlan(ctx, query)
	if err != nil {
		output.StopSpinner(planningSpinner, "Failed to create research plan")
		output.Error("Planning failed", err)
		return "", err
	}

	output.StopSpinner(planningSpinner, "Research plan created")

	// Display the created plan
	output.Info(fmt.Sprintf("Created research plan with %d steps", len(plan.Steps)))
	for i, step := range plan.Steps {
		output.Info(fmt.Sprintf("Step %d: %s (Tool: %s)", i+1, step.Description, step.ToolName))
	}

	// Stage 2: Execute the plan with guided planning
	output.Stage(2, "Executing research with guided planning")

	// Use the planner's GuideAgent method to execute the plan with the agent
	executionSpinner := output.StartSpinner("Executing research plan with guided planning")
	result, err := researcher.Planner.GuideAgent(ctx, researcher.Agent, plan, query)
	if err != nil {
		output.StopSpinner(executionSpinner, "Failed to execute research plan")
		output.Error("Research failed", err)
		return "", err
	}
	output.StopSpinner(executionSpinner, "Research completed")

	// Stage 3: Results
	output.Stage(3, "Research complete")
	output.Result("Web research completed successfully")

	return result, nil
}
