package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent/examples"
	"github.com/theapemachine/errnie"
)

/*
exampleCmd is a command that runs example workflows.
*/
var exampleCmd = &cobra.Command{
	Use:   "example [type]",
	Short: "Run example workflows",
	Long:  `Runs example workflows to demonstrate the agent framework capabilities`,
	Args:  cobra.ArbitraryArgs,
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// Get API key from flags or environment
		apiKey, _ := cmd.Flags().GetString("api-key")
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}

		if apiKey == "" {
			errnie.Error(fmt.Errorf("API key not provided. Use --api-key flag or set OPENAI_API_KEY environment variable"))
			return
		}

		// Common parameters
		topic, _ := cmd.Flags().GetString("topic")
		message, _ := cmd.Flags().GetString("message")
		task, _ := cmd.Flags().GetString("task")
		maxIterations, _ := cmd.Flags().GetInt("iterations")
		timeout, _ := cmd.Flags().GetInt("timeout")

		// If no args provided, display interactive TUI
		if len(args) == 0 {
			err := runExampleTUI(apiKey, topic, message, task, maxIterations, timeout)
			if err != nil {
				errnie.Error(err)

				// Fall back to text list if TUI fails
				fmt.Println("Available example types:")
				fmt.Println("  research       - Research assistant example that can search and synthesize information")
				fmt.Println("  chat           - Simple chat example with an AI assistant")
				fmt.Println("  iteration      - Example demonstrating iterative improvement of a task")
				fmt.Println("  communication  - Example showing communication between multiple agents")
				fmt.Println("  memory         - Example demonstrating memory capabilities")
				fmt.Println("\nUsage:")
				fmt.Println("  caramba example [type] [flags]")
				fmt.Println("\nRun 'caramba example --help' for flags information")
			}
			return
		}

		// Get the example type
		exampleType := args[0]

		// Run the selected example
		var err error
		switch exampleType {
		case "research":
			err = examples.ResearchExample(apiKey, topic)
		case "chat":
			err = examples.ChatExample(apiKey, message)
		case "iteration":
			err = examples.IterationExample(apiKey, task, maxIterations, timeout)
		case "communication":
			err = examples.CommunicationExample(apiKey)
		case "memory":
			err = examples.MemoryExample(apiKey)
		case "browser":
			err = examples.BrowserExample(apiKey, topic)
		default:
			errnie.Error(fmt.Errorf("unknown example type: %s", exampleType))
			fmt.Println("Available examples: research, chat, iteration, communication, memory, browser")
			return
		}

		if err != nil {
			errnie.Error(err)
		}
	},
}

func init() {
	rootCmd.AddCommand(exampleCmd)

	// Add common flags
	exampleCmd.Flags().String("api-key", "", "API key for the LLM provider (or set OPENAI_API_KEY env var)")
	exampleCmd.Flags().String("topic", "artificial intelligence", "Research topic for the research example")
	exampleCmd.Flags().String("message", "Hello, how are you today?", "Message to send for the chat example")
	exampleCmd.Flags().String("task", "Write a short story about a robot learning to understand human emotions.", "Task for the iteration example")
	exampleCmd.Flags().Int("iterations", 3, "Maximum number of iterations for the iteration example")
	exampleCmd.Flags().Int("timeout", 120, "Timeout in seconds for the iteration example")
}
