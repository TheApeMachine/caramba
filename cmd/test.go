package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/process/fractal"
	"github.com/theapemachine/caramba/process/holographic"
	"github.com/theapemachine/caramba/process/quantum"
	"github.com/theapemachine/caramba/process/temporal"
	"github.com/theapemachine/caramba/tools"
)

// testCmd represents the test command
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Used to test the agent",
	Long:  `Executes a test setup`,
	Run: func(cmd *cobra.Command, args []string) {
		pipeline := buildPipeline()

		// Now you can ask the agent to do things like:
		for event := range pipeline.Execute("List all files in the container") {
			// The agent will:
			// 1. Understand the request
			// 2. Generate a JSON command to use the container tool
			// 3. Execute the command
			// 4. Return both its response and the command output
			fmt.Print(event.Content)
		}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}

func buildPipeline() *ai.Pipeline {
	// Create process agents
	temporalAgent := ai.NewAgent()
	temporalAgent.RegisterProcess("temporal", &temporal.Process{})

	holoAgent := ai.NewAgent()
	holoAgent.RegisterProcess("holographic", &holographic.Process{})

	quantumAgent := ai.NewAgent()
	quantumAgent.RegisterProcess("quantum", &quantum.Process{})

	fractalAgent := ai.NewAgent()
	fractalAgent.RegisterProcess("fractal", &fractal.Process{})

	// Create aggregator agent
	aggregatorAgent := ai.NewAgent()

	// Create tool agent
	containerAgent := ai.NewAgent()
	containerAgent.RegisterTool("container", tools.NewContainer())

	pipeline := ai.NewPipeline()

	// Stage 1: Run all processes in parallel
	pipeline.AddStage(true, func(outputs []string) string {
		// Combine all process outputs into a new context
		return "Based on the following analyses:\n" + strings.Join(outputs, "\n")
	}, temporalAgent, holoAgent, quantumAgent, fractalAgent)

	// Stage 2: Aggregate results
	pipeline.AddStage(false, nil, aggregatorAgent)

	// Stage 3: Execute tool based on aggregated context
	pipeline.AddStage(false, nil, containerAgent)

	return pipeline
}
