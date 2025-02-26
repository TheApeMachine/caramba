package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent/examples"
	"github.com/theapemachine/errnie"
)

type ErrorUnknownExample struct {
	ExampleType string
}

func (e *ErrorUnknownExample) Error() string {
	return fmt.Sprintf("unknown example type: %s", e.ExampleType)
}

/*
exampleCmd is a command that runs example workflows.
*/
var exampleCmd = &cobra.Command{
	Use:   "example [type]",
	Short: "Run example workflows",
	Long:  `Runs example workflows to demonstrate the agent framework capabilities`,
	Args:  cobra.ArbitraryArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// Get the example type
		exampleType := args[0]

		switch exampleType {
		case "researcher":
			example := examples.NewResearcher()
			example.Run(cmd.Context(), "How much money did Elvis pay to own the moon?")
		default:
			return &ErrorUnknownExample{ExampleType: exampleType}
		}

		return nil
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
