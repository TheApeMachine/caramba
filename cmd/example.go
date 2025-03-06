package cmd

import (
	"io"
	"os"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/workflow"
)

// Example command variables
var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) error {
			errnie.SetLevel(log.DebugLevel)
			log.Info("Starting example")

			// Create a pipeline with a message, agent, and provider
			pipeline := workflow.NewPipeline(
				core.NewMessage("user", "User", "Tell me a short joke about programming."),
				ai.NewAgent(),
				provider.NewOpenAIProvider("", ""),
			)

			defer pipeline.Close()

			// Copy the pipeline output to stdout
			_, err := io.Copy(os.Stdout, pipeline)
			if err != nil {
				return err
			}

			return nil
		},
	}
)

func init() {
	rootCmd.AddCommand(exampleCmd)
}

var longExample = `
Example demonstrates various capabilities of the Caramba framework.
`
