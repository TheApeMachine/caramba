/*
Package core provides the foundational interfaces and implementations for the Caramba agent framework.
It defines the core abstractions like Agent, LLMProvider, Memory, Tool, and Planner that form
the backbone of the agent system.
*/
package core

import (
	"context"

	"github.com/theapemachine/caramba/pkg/process"
)

/*
Agent represents the interface for an LLM-powered agent.
It defines the core functionality for executing tasks, managing tools,
memory, planning, and communication with other agents.
*/
type Agent interface {
	// Execute runs the agent with the provided input and returns a response
	Execute(context.Context) (string, error)

	// Name returns the name of the agent
	Name() string

	// Memory returns the memory system for the agent
	Memory() Memory

	// IterationLimit returns the iteration limit for the agent
	IterationLimit() int

	// LLM returns the LLM provider for the agent
	LLM() LLMProvider

	// Streaming returns whether the agent is streaming
	Streaming() bool

	// Params returns the parameters for the agent
	Params() *LLMParams

	// SystemPrompt returns the system prompt for the agent
	SystemPrompt() string

	// Status returns the status of the agent
	Status() AgentStatus

	// SetStatus sets the status of the agent
	SetStatus(status AgentStatus)

	// AddTool adds a new tool to the agent
	AddTool(tool Tool)

	// AddUserMessage adds a user message to the agent
	AddUserMessage(message string)

	// AddAssistantMessage adds an assistant message to the agent
	AddAssistantMessage(message string)

	// SetMemory sets the memory system for the agent
	SetMemory(memory Memory)

	// SetPlanner sets the planner for the agent
	SetPlanner(planner Agent)

	// SetOptimizer sets the optimizer for the agent
	SetOptimizer(optimizer Agent)

	// SetLLM sets the LLM provider for the agent
	SetLLM(llm LLMProvider)

	// SetSystemPrompt sets the system prompt for the agent
	SetSystemPrompt(prompt string)

	// SetProcess sets the process for the agent
	SetProcess(process process.StructuredOutput)

	// SetIterationLimit sets the iteration limit for the agent
	SetIterationLimit(limit int)

	// GetTool returns a tool by name.
	GetTool(name string) Tool

	// SetStreaming sets the streaming mode for the agent
	SetStreaming(streaming bool)
}

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
	Schema                    any
}

type ResponseType string

const (
	ResponseTypeContent  ResponseType = "content"
	ResponseTypeToolCall ResponseType = "tool_call"
	ResponseTypeError    ResponseType = "error"
)

type LLMResponse struct {
	Type      ResponseType
	Model     string
	Content   string
	ToolCalls []ToolCall
	Refusal   string
	Error     error
}

type ToolCall struct {
	Name string
	Args map[string]any
}

/*
Memory defines the interface for the agent's memory system.
It provides methods for storing, retrieving, and searching information
that the agent needs to persist across interactions.
*/
type Memory interface {
	// QueryAgent returns the query agent for the memory
	QueryAgent() Agent

	// MutateAgent returns the mutate agent for the memory
	MutateAgent() Agent

	// Query the memory
	Query(context.Context, *process.MemoryLookup) (string, error)

	// Mutate the memory
	Mutate(context.Context, *process.MemoryMutate) error
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
	Value any
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
	Execute(ctx context.Context, args map[string]any) (any, error)

	// Schema returns the JSON schema for the tool's arguments
	Schema() map[string]any
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

	// GuideAgent executes a plan with an agent, guiding it through each step
	GuideAgent(ctx context.Context, agent Agent, plan Plan, query string) (string, error)
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
	Arguments map[string]any
}

/*
Workflow defines the interface for a workflow.
Workflows allow for creating sequences of tool executions with
conditional logic and error handling.
*/
type Workflow interface {
	// AddStep adds a step to the workflow
	AddStep(name string, tool Tool, args map[string]any) Workflow

	// AddConditionalStep adds a conditional step to the workflow
	AddConditionalStep(name string, condition string, tool Tool, args map[string]any) Workflow

	// Execute executes the workflow with the given input
	Execute(ctx context.Context, input map[string]any) (map[string]any, error)

	// SetErrorHandler sets a handler for errors
	SetErrorHandler(handler func(error) error) Workflow
}
