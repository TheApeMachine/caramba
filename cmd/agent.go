package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

/*
agentCmd is a command that creates and runs an agent with various configurations.
It supports different agent types and allows passing input text for the agent to process.
*/
var agentCmd = &cobra.Command{
	Use:   "agent",
	Short: "Create and run an agent",
	Long:  `Creates an agent and runs it with the provided input`,
	Run: func(cmd *cobra.Command, args []string) {
		// Initialize logger
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// Get flags
		name, _ := cmd.Flags().GetString("name")
		agentType, _ := cmd.Flags().GetString("type")
		apiKey, _ := cmd.Flags().GetString("api-key")
		input, _ := cmd.Flags().GetString("input")

		// Create the agent factory
		factory := agent.NewAgentFactory()

		// Create the agent based on the type
		var a interface{}

		switch strings.ToLower(agentType) {
		case "basic":
			a = factory.CreateBasicAgent(name, apiKey)
		case "research":
			searchAPIKey, _ := cmd.Flags().GetString("search-api-key")
			searchID, _ := cmd.Flags().GetString("search-id")
			a = factory.CreateResearchAgent(name, apiKey, searchAPIKey, searchID)
		default:
			errnie.Error(fmt.Errorf("unknown agent type: %s", agentType))
			return
		}

		// Execute the agent
		result, err := a.(core.Agent).Execute(context.Background(), input)
		if err != nil {
			errnie.Error(err)
			return
		}

		// Print the result
		fmt.Println("Agent Response:")
		fmt.Println(result)
	},
}

/*
init initializes the agent command by adding it to the root command and setting up flags.
It defines all the necessary flags for configuring different types of agents and marks
required flags to ensure proper command usage.
*/
func init() {
	rootCmd.AddCommand(agentCmd)

	// Add flags
	agentCmd.Flags().StringP("name", "n", "Agent", "Name of the agent")
	agentCmd.Flags().StringP("type", "t", "basic", "Type of agent (basic, research)")
	agentCmd.Flags().StringP("api-key", "k", "", "API key for the LLM provider")
	agentCmd.Flags().StringP("input", "i", "", "Input text for the agent")
	agentCmd.Flags().String("search-api-key", "", "API key for search (required for research agent)")
	agentCmd.Flags().String("search-id", "", "Search engine ID (required for research agent)")

	// Mark required flags
	agentCmd.MarkFlagRequired("api-key")
	agentCmd.MarkFlagRequired("input")
}
