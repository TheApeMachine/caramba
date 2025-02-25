package core

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/theapemachine/errnie"
)

// IterationResult represents the result of a single iteration
type IterationResult struct {
	Response      string
	ToolResults   map[string]interface{}
	IsComplete    bool
	IterationTime time.Duration
}

// IterationOptions configures the behavior of the Iterator
type IterationOptions struct {
	MaxIterations    int           // Maximum number of iterations (default: 5)
	Timeout          time.Duration // Overall timeout for all iterations (default: 2 minutes)
	CompletionPhrase string        // Phrase that indicates the agent is done (default: "ITERATION_COMPLETE")
}

// DefaultIterationOptions returns the default options for iteration
func DefaultIterationOptions() *IterationOptions {
	return &IterationOptions{
		MaxIterations:    5,
		Timeout:          2 * time.Minute,
		CompletionPhrase: "ITERATION_COMPLETE",
	}
}

// Iterator manages the iteration loop for an agent
type Iterator struct {
	options       *IterationOptions
	history       []IterationResult
	contextPrefix string
	contextSuffix string
}

// NewIterator creates a new Iterator with the given options
func NewIterator(options *IterationOptions) *Iterator {
	if options == nil {
		options = DefaultIterationOptions()
	}

	return &Iterator{
		options: options,
		history: make([]IterationResult, 0),
		contextPrefix: `
You are now entering an iteration loop. You'll have multiple opportunities to reflect on and improve your work.
Your goal is to produce the best possible result through self-reflection and iteration.

When you're satisfied with your response, include the phrase "ITERATION_COMPLETE" at the end.
`,
		contextSuffix: `
Now reflect on your work so far. What could be improved? Are there any mistakes or oversights?
Consider how you can make your response more comprehensive, accurate, and helpful.
`,
	}
}

// SetCompletionPhrase sets a custom completion phrase
func (i *Iterator) SetCompletionPhrase(phrase string) {
	i.options.CompletionPhrase = phrase
}

// SetContextWrapper sets custom prefix and suffix for iteration context
func (i *Iterator) SetContextWrapper(prefix, suffix string) {
	i.contextPrefix = prefix
	i.contextSuffix = suffix
}

// Run executes the iteration loop using the provided agent and input
func (i *Iterator) Run(ctx context.Context, agent Agent, input string) (string, error) {
	// Create a timeout context for the entire iteration process
	timeoutCtx, cancel := context.WithTimeout(ctx, i.options.Timeout)
	defer cancel()

	// Initialize the current context with the original input
	currentContext := i.contextPrefix + "\n\nOriginal input:\n" + input

	// Start the iteration loop
	for iteration := 0; iteration < i.options.MaxIterations; iteration++ {
		// Check for timeout
		if timeoutCtx.Err() != nil {
			errnie.Info("Iteration timeout reached")
			break
		}

		errnie.Info(fmt.Sprintf("Starting iteration %d", iteration+1))

		// Execute the agent with the current context
		startTime := time.Now()
		response, err := agent.Execute(timeoutCtx, currentContext)
		iterationTime := time.Since(startTime)

		if err != nil {
			return "", fmt.Errorf("agent execution failed at iteration %d: %w", iteration+1, err)
		}

		// Check if the agent indicates completion
		isComplete := strings.Contains(response, i.options.CompletionPhrase)
		if isComplete {
			// Remove the completion phrase from the response
			response = strings.ReplaceAll(response, i.options.CompletionPhrase, "")
			response = strings.TrimSpace(response)
		}

		// Process any tool calls in the response
		var toolResults map[string]interface{}
		if baseAgent, ok := agent.(*BaseAgent); ok {
			toolResults, err = baseAgent.GetToolResults(timeoutCtx, response)
			if err != nil {
				errnie.Info("Error processing tool calls: " + err.Error())
				// Continue even if there's an error with tool calls
				toolResults = make(map[string]interface{})
			}

			// If we have tool results, log them
			if len(toolResults) > 0 {
				errnie.Info(fmt.Sprintf("Processed %d tool calls in iteration %d", len(toolResults), iteration+1))
			}
		} else {
			// If the agent doesn't implement GetToolResults, just use an empty map
			toolResults = make(map[string]interface{})
		}

		// Store the result
		result := IterationResult{
			Response:      response,
			ToolResults:   toolResults,
			IsComplete:    isComplete,
			IterationTime: iterationTime,
		}
		i.history = append(i.history, result)

		// If the agent indicates completion, break the loop
		if isComplete {
			errnie.Info("Agent indicated completion at iteration " + fmt.Sprintf("%d", iteration+1))
			break
		}

		// Update the context for the next iteration
		currentContext = i.buildNextIterationContext(input, iteration+1)
	}

	// Return the final response
	if len(i.history) == 0 {
		return "", fmt.Errorf("no iterations completed")
	}

	return i.history[len(i.history)-1].Response, nil
}

// GetHistory returns the iteration history
func (i *Iterator) GetHistory() []IterationResult {
	return i.history
}

// buildNextIterationContext constructs the context for the next iteration
func (i *Iterator) buildNextIterationContext(originalInput string, currentIteration int) string {
	var sb strings.Builder

	// Add the prefix
	sb.WriteString(i.contextPrefix)
	sb.WriteString("\n\n")

	// Add the original input
	sb.WriteString("Original input:\n")
	sb.WriteString(originalInput)
	sb.WriteString("\n\n")

	// Add the iteration history
	sb.WriteString("Previous iterations:\n")

	for idx, result := range i.history {
		sb.WriteString(fmt.Sprintf("--- Iteration %d ---\n", idx+1))
		sb.WriteString(result.Response)
		sb.WriteString("\n\n")

		// Add tool results if they exist
		if len(result.ToolResults) > 0 {
			sb.WriteString("Tool results:\n")
			for toolName, toolResult := range result.ToolResults {
				sb.WriteString(fmt.Sprintf("%s: %v\n", toolName, toolResult))
			}
			sb.WriteString("\n")
		}
	}

	// Add the current iteration number
	sb.WriteString(fmt.Sprintf("--- Current Iteration %d ---\n", currentIteration))

	// Add the suffix (instructions for reflection)
	sb.WriteString(i.contextSuffix)

	return sb.String()
}
