package examples

import (
	"context"
	"fmt"
	"os"

	"github.com/theapemachine/caramba/pkg/api/ai"
	"github.com/theapemachine/caramba/pkg/api/provider"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// CapnpExample demonstrates using Cap'n Proto interfaces
type CapnpExample struct {
	agent ai.Agent
}

// NewCapnp creates a new Cap'n Proto example
func NewCapnp() *CapnpExample {
	errnie.Debug("examples.NewCapnp")

	prvdr := provider.NewProvider(os.Getenv("OPENAI_API_KEY"))
	agent, err := ai.NewAgent(prvdr)

	if err != nil {
		errnie.Error(err)
		return nil
	}

	return &CapnpExample{
		agent: agent,
	}
}

func (example *CapnpExample) Run() (err error) {
	errnie.Info("Starting Cap'n Proto example")

	// Start a new conversation
	params, err := provider.NewConversation("Tell me about the benefits of using Cap'n Proto for serialization")

	if err != nil {
		return errnie.Error(err)
	}

	// Send the message and get the response
	params, err = ai.Ask(context.Background(), example.agent, params)

	if err != nil {
		return errnie.Error(err)
	}

	lastMsg, err := provider.GetLastMessage(params)

	if err != nil {
		return errnie.Error(err)
	}

	fmt.Println(lastMsg.Content())

	// Add a follow-up question
	if err := provider.AddUserMessage(params, "How does Cap'n Proto compare to Protocol Buffers?"); err != nil {
		return errnie.Error(err)
	}

	// Send the follow-up and get the final response
	if params, err = ai.Ask(context.Background(), example.agent, params); err != nil {
		return errnie.Error(err)
	}

	if lastMsg, err = provider.GetLastMessage(params); err != nil {
		return errnie.Error(err)
	}

	fmt.Println(lastMsg.Content())

	return nil
}
