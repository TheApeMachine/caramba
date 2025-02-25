package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/errnie"
)

// IterationExample demonstrates how agents can iterate, reflect, and improve their responses
func IterationExample(apiKey, task string, maxIterations int, timeoutSeconds int) error {
	if task == "" {
		task = "Write a short story about a robot learning to understand human emotions."
	}

	if maxIterations <= 0 {
		maxIterations = 3
	}

	if timeoutSeconds <= 0 {
		timeoutSeconds = 120
	}

	// Create an agent using OpenAI provider
	errnie.Info("🤖 Creating agent with iteration capability")

	// Create the primary provider
	provider := llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")

	// Create an agent using the builder pattern
	agent := core.NewAgentBuilder("IterativeAgent").
		WithLLM(provider).
		Build()

	// Set up iteration options
	iterOptions := &core.IterationOptions{
		MaxIterations:    maxIterations,
		Timeout:          time.Duration(timeoutSeconds) * time.Second,
		CompletionPhrase: "ITERATION_COMPLETE",
	}

	fmt.Printf("Starting iterative task: %s\n", task)
	fmt.Printf("Maximum iterations: %d\n", maxIterations)
	fmt.Printf("Timeout: %d seconds\n", timeoutSeconds)
	fmt.Println("---")

	// Start a timer
	startTime := time.Now()

	// Execute the agent with iteration
	result, err := agent.ExecuteWithIteration(context.Background(), task, iterOptions)
	if err != nil {
		errnie.Error(err)
		return err
	}

	// Calculate the total time
	duration := time.Since(startTime)

	fmt.Println("===== FINAL RESULT =====")
	fmt.Println(result)
	fmt.Println("========================")
	fmt.Printf("Task completed in %v\n", duration)

	return nil
}
