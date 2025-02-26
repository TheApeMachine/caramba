package examples

import (
	"context"
	"os"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/errnie"
)

/*
Researcher is an example agent that uses the browser tool.
*/
type Researcher struct {
	Agent core.Agent
}

func NewResearcher() *Researcher {
	return &Researcher{
		Agent: core.NewAgentBuilder(
			"BrowserAgent",
		).WithLLM(
			llm.NewOpenAIProvider(os.Getenv("OPENAI_API_KEY"), "gpt-4o-mini"),
		).WithSystemPrompt(
			"You are a web researcher that can use the browser tool to search the web.",
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

	response, err := researcher.Agent.Execute(ctx, core.LLMMessage{
		Role:    "user",
		Content: query,
	})

	if err != nil {
		errnie.Error(err)
		return "", err
	}

	return response, nil
}
