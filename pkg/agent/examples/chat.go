package examples

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/errnie"
)

// ChatExample runs a simple chat interaction with an agent
func ChatExample(apiKey, message string) error {
	if message == "" {
		message = "Hello, how are you today?"
	}

	// Create the agent factory
	factory := agent.NewAgentFactory()

	// Create a basic agent
	agent := factory.CreateBasicAgent("ChatBot", apiKey)

	// Execute the agent
	fmt.Printf("User: %s\n", message)
	response, err := agent.Execute(context.Background(), message)
	if err != nil {
		errnie.Error(err)
		return err
	}

	fmt.Printf("Agent: %s\n", response)
	return nil
}
