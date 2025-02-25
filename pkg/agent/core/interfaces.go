/*
Package core provides the foundational interfaces and implementations for the Caramba agent framework.
It defines the core abstractions like Agent, LLMProvider, Memory, Tool, and Planner that form
the backbone of the agent system.
*/
package core

import (
	"context"
)

/*
LLMProvider defines the interface for language model providers.
It abstracts the interaction with various LLM services and allows
for a consistent API regardless of the underlying provider.
*/
type LLMProvider interface {
	// GenerateResponse generates a response from the LLM
	GenerateResponse(context.Context, LLMParams) LLMResponse

	// StreamResponse generates a response from the LLM and streams it
	StreamResponse(context.Context, LLMParams) <-chan LLMResponse

	// Name returns the name of the LLM provider
	Name() string
}

type LLMMessage struct {
	Role    string
	Content string
}

func SystemMessage(content string) LLMMessage {
	return LLMMessage{
		Role:    "system",
		Content: content,
	}
}

type LLMParams struct {
	Messages                  []LLMMessage
	Model                     string
	MaxTokens                 int
	TopP                      float64
	Temperature               float64
	FrequencyPenalty          float64
	PresencePenalty           float64
	StopSequences             []string
	SystemPrompt              string
	Tools                     []Tool
	ResponseFormatName        string
	ResponseFormatDescription string
	Schema                    interface{}
}

type LLMResponse struct {
	Content   string
	ToolCalls []ToolCall
	Refusal   string
	Error     error
}

/*
Memory defines the interface for the agent's memory system.
It provides methods for storing, retrieving, and searching information
that the agent needs to persist across interactions.
*/
type Memory interface {
	// Store stores a key-value pair in memory
	Store(ctx context.Context, key string, value interface{}) error

	// Retrieve retrieves a value from memory by key
	Retrieve(ctx context.Context, key string) (interface{}, error)

	// Search searches the memory using a query
	Search(ctx context.Context, query string, limit int) ([]MemoryEntry, error)

	// Clear clears the memory
	Clear(ctx context.Context) error
}

/*
MemoryEnhancer defines the interface for memory systems that can enhance prompts.
It provides methods to prepare context with relevant memories for a given query.
*/
type MemoryEnhancer interface {
	// PrepareContext enriches a prompt with relevant memories
	PrepareContext(ctx context.Context, agentID string, query string) (string, error)
}

/*
MemoryExtractor defines the interface for memory systems that can extract memories.
It provides methods to extract important information from text to be stored as memories.
*/
type MemoryExtractor interface {
	// ExtractMemories extracts important information from text that should be remembered
	ExtractMemories(ctx context.Context, agentID string, text string, source string) ([]string, error)
}

/*
MemoryEntry represents an entry in the memory system.
It includes the key, value, and a relevance score for search results.
*/
type MemoryEntry struct {
	/* Key is the identifier for the memory entry */
	Key string
	/* Value is the stored data */
	Value interface{}
	/* Score indicates the relevance of this entry to a search query */
	Score float64
}

/*
Tool defines the interface for agent tools.
Tools extend an agent's capabilities by allowing it to perform
specific actions like web searches, calculations, or API calls.
*/
type Tool interface {
	// Name returns the name of the tool
	Name() string

	// Description returns the description of the tool
	Description() string

	// Execute executes the tool with the given arguments
	Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)

	// Schema returns the JSON schema for the tool's arguments
	Schema() map[string]interface{}
}

/*
Planner defines the interface for the agent's planning component.
It enables the agent to break down complex tasks into manageable steps
and execute them in sequence.
*/
type Planner interface {
	// CreatePlan creates a plan based on the input
	CreatePlan(ctx context.Context, input string) (Plan, error)

	// ExecutePlan executes a plan and returns the result
	ExecutePlan(ctx context.Context, plan Plan) (string, error)
}

/*
Plan represents a plan created by a planner.
It consists of a sequence of steps to be executed.
*/
type Plan struct {
	/* Steps is the ordered list of steps in the plan */
	Steps []PlanStep
}

/*
PlanStep represents a step in a plan.
It includes a description of the step, the tool to use, and the arguments for that tool.
*/
type PlanStep struct {
	/* Description explains what this step is supposed to accomplish */
	Description string
	/* ToolName identifies which tool should be used for this step */
	ToolName string
	/* Arguments contains the parameters to pass to the tool */
	Arguments map[string]interface{}
}

/*
Workflow defines the interface for a workflow.
Workflows allow for creating sequences of tool executions with
conditional logic and error handling.
*/
type Workflow interface {
	// AddStep adds a step to the workflow
	AddStep(name string, tool Tool, args map[string]interface{}) Workflow

	// AddConditionalStep adds a conditional step to the workflow
	AddConditionalStep(name string, condition string, tool Tool, args map[string]interface{}) Workflow

	// Execute executes the workflow with the given input
	Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

	// SetErrorHandler sets a handler for errors
	SetErrorHandler(handler func(error) error) Workflow
}
