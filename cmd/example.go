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
	Args:  cobra.MinimumNArgs(1),
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

		// Get the example type
		exampleType := args[0]

		// Common parameters
		topic, _ := cmd.Flags().GetString("topic")
		message, _ := cmd.Flags().GetString("message")
		task, _ := cmd.Flags().GetString("task")
		maxIterations, _ := cmd.Flags().GetInt("iterations")
		timeout, _ := cmd.Flags().GetInt("timeout")

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
		default:
			errnie.Error(fmt.Errorf("unknown example type: %s", exampleType))
			fmt.Println("Available examples: research, chat, iteration, communication, memory")
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
