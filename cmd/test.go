package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/tools"
)

// testCmd represents the test command
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Used to test the agent",
	Long:  `Executes a test setup`,
	Run: func(cmd *cobra.Command, args []string) {
		pipeline := buildPipeline()
		for event := range pipeline.Execute("Develop a simple web server") {
			fmt.Print(event.Content)
		}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}

func buildPipeline() *ai.Pipeline {
	// temporalAgent := ai.NewAgent("reasoner", &temporal.Process{})
	// holoAgent := ai.NewAgent("reasoner", &holographic.Process{})
	// quantumAgent := ai.NewAgent("reasoner", &quantum.Process{})
	// fractalAgent := ai.NewAgent("reasoner", &fractal.Process{})
	containerAgent := ai.NewAgent("developer", tools.NewContainer())

	pipeline := ai.NewPipeline()

	pipeline.AddSequentialStage(
		nil,
		containerAgent,
	)

	return pipeline
}
