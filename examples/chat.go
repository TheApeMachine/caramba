package examples

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

/*
Chat shows off the basic chat example, where the user goes back and forth
with an AI agent, that keeps track of the conversation in a thread.
We will also add the browser tool to the agent, so that it can browse the web.
*/
type Chat struct {
	ctx   context.Context
	agent *ai.Agent
}

/*
NewChat creates a new Chat instance with the specified context and role.
It initializes an empty scratchpad for accumulating assistant responses
and sets up the basic formatting configuration.

Parameters:
  - ctx: The context for operations
  - role: The role designation for the AI agent
*/
func NewChat(ctx context.Context, role string) *Chat {
	return &Chat{
		ctx:   ctx,
		agent: ai.NewAgent(ctx, role, 1),
	}
}

/*
Run starts the chat loop, which allows the user to interact with the agent.
It adds the browser tool to the agent, so that it can browse the web.
*/
func (chat *Chat) Run() error {
	chat.agent.AddTools(
		tools.NewBrowser(),
	)

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")

		if !scanner.Scan() {
			break
		}

		input := scanner.Text()
		if strings.ToLower(input) == "exit" {
			break
		}

		for event := range chat.agent.Generate(
			chat.ctx,
			provider.NewMessage(provider.RoleUser, input),
		) {
			switch event.Type() {
			case "chunk":
				if data, ok := event.Data().(map[string]interface{}); ok {
					if text, ok := data["text"].(string); ok {
						fmt.Print(text)
					}
				}
			case "error":
				if data, ok := event.Data().(map[string]interface{}); ok {
					if err, ok := data["error"].(error); ok {
						return errnie.Error(err)
					}
				}
				return errnie.Error(fmt.Errorf("unknown error"))
			}
		}
	}

	return nil
}
