package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/errnie"
)

// CommunicationExample demonstrates how agents can communicate with each other
func CommunicationExample(apiKey string) error {
	fmt.Println("Creating agents for communication demonstration...")

	// Create multiple agents
	agent1 := core.NewAgentBuilder("Alice").
		WithLLM(llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")).
		Build()

	agent2 := core.NewAgentBuilder("Bob").
		WithLLM(llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")).
		Build()

	agent3 := core.NewAgentBuilder("Charlie").
		WithLLM(llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")).
		Build()

	// Get messengers
	messenger1 := agent1.GetMessenger()
	messenger2 := agent2.GetMessenger()
	messenger3 := agent3.GetMessenger()

	// Direct messaging
	fmt.Println("\n=== Direct Messaging ===")
	messageID, err := messenger1.SendDirect(context.Background(), "Bob", "Hello Bob, this is Alice!", core.MessageTypeText, nil)
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Printf("Alice sent a direct message to Bob (ID: %s)\n", messageID)
	}

	// Create a topic
	fmt.Println("\n=== Topic Creation ===")
	err = messenger1.CreateTopic(context.Background(), "general", "General discussion channel")
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Println("Alice created the 'general' topic")
	}

	// Subscribe to topics
	fmt.Println("\n=== Topic Subscription ===")
	err = messenger2.Subscribe(context.Background(), "general")
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Println("Bob subscribed to the 'general' topic")
	}

	err = messenger3.Subscribe(context.Background(), "general")
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Println("Charlie subscribed to the 'general' topic")
	}

	// Publish to topic
	fmt.Println("\n=== Topic Publishing ===")
	messageID, err = messenger1.Publish(context.Background(), "general", "Welcome to the general discussion!", core.MessageTypeText, nil)
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Printf("Alice published a message to the 'general' topic (ID: %s)\n", messageID)
	}

	// Broadcasting
	fmt.Println("\n=== Broadcasting ===")
	messageIDs, err := messenger3.Broadcast(context.Background(), "Attention everyone!", core.MessageTypeSystem, nil)
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Printf("Charlie broadcast a message to all agents (ID: %s)\n", messageIDs[0])
	}

	// Retrieve messages
	fmt.Println("\n=== Message Retrieval ===")
	time.Sleep(100 * time.Millisecond) // Give time for message processing

	// Get Bob's messages
	messages, err := messenger2.GetMessages(context.Background(), time.Now().Add(-time.Hour))
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Printf("Bob has %d messages\n", len(messages))
		for _, msg := range messages {
			fmt.Printf("  From: %s, Content: %s\n", msg.Sender, msg.Content)
		}
	}

	// Get Charlie's messages
	messages, err = messenger3.GetMessages(context.Background(), time.Now().Add(-time.Hour))
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Printf("Charlie has %d messages\n", len(messages))
		for _, msg := range messages {
			fmt.Printf("  From: %s, Content: %s\n", msg.Sender, msg.Content)
		}
	}

	// Get topics
	fmt.Println("\n=== List Topics ===")
	topics, err := messenger1.GetTopics(context.Background())
	if err != nil {
		errnie.Error(err)
	} else {
		fmt.Printf("Available topics: %d\n", len(topics))
		for _, topic := range topics {
			fmt.Printf("  %s: %s (Created by: %s, Subscribers: %d)\n",
				topic.Name, topic.Description, topic.Creator, len(topic.Subscribers))
		}
	}

	return nil
}
