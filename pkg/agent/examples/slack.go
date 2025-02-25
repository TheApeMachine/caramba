package examples

import (
	"context"
	"fmt"
	"os"

	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/agent/tools"
)

// RunSlackExample demonstrates the usage of the Slack tool
func RunSlackExample() {
	// Get Slack token from environment
	slackToken := os.Getenv("SLACK_TOKEN")
	if slackToken == "" {
		fmt.Println("SLACK_TOKEN environment variable not set")
		return
	}

	// Default channel to use
	defaultChannel := os.Getenv("SLACK_CHANNEL")
	if defaultChannel == "" {
		fmt.Println("SLACK_CHANNEL environment variable not set, using general")
		defaultChannel = "general"
	}

	// Create a new Slack tool
	slackTool := tools.NewSlackTool(slackToken, defaultChannel)

	// Context for all operations
	ctx := context.Background()

	// Example 1: List channels
	fmt.Println("Listing channels...")
	result, err := slackTool.Execute(ctx, map[string]interface{}{
		"action": "list_channels",
		"limit":  10,
	})
	if err != nil {
		fmt.Printf("Error listing channels: %v\n", err)
	} else {
		prettyPrint("Channels", result)
	}

	// Example 2: Post a message
	fmt.Println("\nPosting a message...")
	result, err = slackTool.Execute(ctx, map[string]interface{}{
		"action":  "post_message",
		"channel": defaultChannel,
		"text":    "Hello from the Caramba Slack tool!",
	})
	if err != nil {
		fmt.Printf("Error posting message: %v\n", err)
	} else {
		prettyPrint("Message", result)

		// Store the timestamp for the next examples
		msgTimestamp := result.(map[string]interface{})["timestamp"].(string)

		// Example 3: Add a reaction to the message
		fmt.Println("\nAdding a reaction...")
		result, err = slackTool.Execute(ctx, map[string]interface{}{
			"action":    "add_reaction",
			"channel":   defaultChannel,
			"timestamp": msgTimestamp,
			"reaction":  "thumbsup",
		})
		if err != nil {
			fmt.Printf("Error adding reaction: %v\n", err)
		} else {
			prettyPrint("Reaction", result)
		}

		// Example 4: Update the message
		fmt.Println("\nUpdating the message...")
		result, err = slackTool.Execute(ctx, map[string]interface{}{
			"action":    "update_message",
			"channel":   defaultChannel,
			"timestamp": msgTimestamp,
			"text":      "Updated message from the Caramba Slack tool!",
		})
		if err != nil {
			fmt.Printf("Error updating message: %v\n", err)
		} else {
			prettyPrint("Updated Message", result)
		}
	}

	// Example 5: Get channel history
	fmt.Println("\nGetting channel history...")
	result, err = slackTool.Execute(ctx, map[string]interface{}{
		"action":  "get_channel_history",
		"channel": defaultChannel,
		"limit":   5,
	})
	if err != nil {
		fmt.Printf("Error getting channel history: %v\n", err)
	} else {
		prettyPrint("Channel History", result)
	}

	// Example 6: Search messages
	fmt.Println("\nSearching messages...")
	result, err = slackTool.Execute(ctx, map[string]interface{}{
		"action": "search_messages",
		"query":  "Caramba",
		"limit":  5,
	})
	if err != nil {
		fmt.Printf("Error searching messages: %v\n", err)
	} else {
		prettyPrint("Search Results", result)
	}

	// Example 7: Using the AgentFactory
	fmt.Println("\nCreating a Slack agent using AgentFactory...")
	factory := agent.NewAgentFactory()
	slackAgent := factory.CreateSlackAgent("SlackBot", "your-llm-api-key", slackToken, defaultChannel)
	fmt.Println("Slack agent created successfully with name 'SlackBot'")
	fmt.Println("The agent has the Slack tool available")

	// Use the agent
	_, err = slackAgent.Execute(ctx, "Send a message to the Slack channel saying 'Hello from the Caramba agent!'")
	if err != nil {
		fmt.Printf("Error executing agent: %v\n", err)
	}
}
