package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/caramba/pkg/agent/workflow"
	"github.com/theapemachine/errnie"
)

// ResearchExample runs an example workflow for research tasks
func ResearchExample(apiKey, topic string) error {
	if topic == "" {
		topic = "artificial intelligence"
	}

	// Create an LLM provider
	llmProvider := llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")

	// Create a memory store
	memoryStore := memory.NewInMemoryStore()

	// Create tools
	calculator := tools.NewCalculator()
	webSearch := tools.NewWebSearch("", "") // Using mock search for demo
	formatter := tools.NewFormatter()       // Add formatter tool for text processing

	// Create a planner
	tools := map[string]core.Tool{
		calculator.Name(): calculator,
		webSearch.Name():  webSearch,
		formatter.Name():  formatter,
	}
	planner := core.NewSimplePlanner(llmProvider, tools)

	// Create an agent
	agent := core.NewAgentBuilder("ResearchAgent").
		WithLLM(llmProvider).
		WithMemory(memoryStore).
		WithTool(calculator).
		WithTool(webSearch).
		WithTool(formatter). // Add formatter tool to agent
		WithPlanner(planner).
		Build()

	// Store the start time in memory
	_ = memoryStore.Store(context.Background(), "start_time", time.Now())

	// Create a research workflow
	wf := workflow.NewWorkflow().
		AddStep("search", webSearch, map[string]interface{}{
			"query":       topic,
			"max_results": 3,
		}).
		AddStep("analyze", formatter, map[string]interface{}{
			"template": "Analysis of search results on {{.topic}}:\n\n{{.search}}",
		}).
		AddStep("summarize", formatter, map[string]interface{}{
			"template": "Summary of research on {{.topic}}:\n\n{{.analyze}}",
		})

	fmt.Printf("Starting research on: %s\n", topic)

	// Run the workflow
	results, err := agent.RunWorkflow(context.Background(), wf, map[string]interface{}{
		"topic": topic,
	})

	if err != nil {
		errnie.Error(err)
		return err
	}

	// Retrieve the start time from memory
	startTimeObj, err := memoryStore.Retrieve(context.Background(), "start_time")
	if err != nil {
		errnie.Error(err)
	}

	startTime, ok := startTimeObj.(time.Time)
	if !ok {
		errnie.Error(fmt.Errorf("failed to retrieve start time"))
	}

	// Print the results
	fmt.Println("Research Results:")
	fmt.Println("================")

	for k, v := range results {
		fmt.Printf("%s: %v\n", k, v)
	}

	if ok {
		fmt.Printf("\nResearch completed in: %v\n", time.Since(startTime))
	}

	return nil
}
