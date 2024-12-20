package ai

import (
	"io"
	"strings"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

type Agent struct {
	provider provider.Provider
	tools    map[string]Tool
	process  Process
	prompt   *Prompt
	buf      strings.Builder
	buffer   *Buffer
}

func NewAgent() *Agent {
	agent := &Agent{
		provider: provider.NewBalancedProvider(),
		tools:    make(map[string]Tool),
		prompt:   NewPrompt(),
		buffer:   NewBuffer(),
	}

	// Set system prompt once during initialization
	agent.buffer.Poke(provider.Message{
		Role:    "system",
		Content: agent.prompt.Build(),
	})

	return agent
}

func (agent *Agent) RegisterProcess(name string, process Process) {
	agent.process = process
	agent.prompt.WithRole("process").WithSchema(process.GenerateSchema())
}

func (agent *Agent) RegisterTool(name string, tool Tool) {
	agent.tools[name] = tool
	agent.prompt.WithRole("tool").WithSchema(tool.GenerateSchema())
}

// Implement io.ReadWriteCloser
func (agent *Agent) Write(p []byte) (n int, err error) {
	// Container output becomes the prompt
	for event := range agent.Generate(string(p)) {
		if event.Type == provider.EventToken {
			agent.buf.WriteString(event.Content)
		}
	}
	return len(p), nil
}

func (agent *Agent) Read(p []byte) (n int, err error) {
	// Return accumulated response
	if agent.buf.Len() == 0 {
		return 0, io.EOF
	}

	n = copy(p, agent.buf.String())
	agent.buf.Reset()
	return n, nil
}

func (agent *Agent) Close() error {
	return nil
}

func (agent *Agent) Generate(prompt string) <-chan provider.Event {
	agent.buffer.Poke(provider.Message{
		Role:    "user",
		Content: prompt,
	})

	params := provider.GenerationParams{
		Messages: agent.buffer.Peek(),
	}

	out := make(chan provider.Event)
	go func() {
		defer close(out)

		var response strings.Builder

		// Stream events from provider
		for event := range agent.provider.Generate(params) {
			// Always stream non-token events immediately
			if event.Type != provider.EventToken {
				out <- event
				continue
			}

			response.WriteString(event.Content)
			out <- event
		}

		// Store the complete response
		responseStr := response.String()
		agent.buffer.Poke(provider.Message{
			Role:    "assistant",
			Content: responseStr,
		})

		// Handle tool usage if we have tools
		if len(agent.tools) > 0 {
			// Try to extract JSON from the response
			blocks := utils.ExtractJSONBlocks(responseStr)
			if len(blocks) > 0 {
				// Use the first valid tool command
				for _, block := range blocks {
					if toolOutput := agent.tryExecuteTool(block); toolOutput != "" {
						out <- provider.Event{
							Type:    provider.EventToken,
							Content: toolOutput,
						}
						break
					}
				}
			}
		}

		out <- provider.Event{Type: provider.EventDone}
	}()

	return out
}

// tryExecuteTool attempts to execute a tool command from a JSON block
func (agent *Agent) tryExecuteTool(block interface{}) string {
	// Try to convert block to map
	params, ok := block.(map[string]interface{})
	if !ok {
		return ""
	}

	// Look for tool name in params
	toolName, _ := params["tool"].(string)
	if tool, exists := agent.tools[toolName]; exists {
		// Check if tool needs connection and hasn't been connected yet
		if _, ok := tool.(io.ReadWriteCloser); ok {
			// Connect the tool to the agent itself as the ReadWriteCloser
			tool.Connect(agent)
		}
		return tool.Use(params)
	}

	return ""
}
