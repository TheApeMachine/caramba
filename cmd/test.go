package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/process/review"
	"github.com/theapemachine/caramba/tools"
)

// testCmd represents the test command
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Used to test the agent",
	Long:  `Executes a test setup`,
	Run: func(cmd *cobra.Command, args []string) {
		pipeline := buildPipeline()
		for event := range pipeline.Execute("What can you tell me about Fan Factory? This is their website, if it helps: https://www.fanfactory.nl") {
			fmt.Print(event.Content)
		}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}

func buildPipeline() *ai.Pipeline {
	pipeline := ai.NewPipeline()

	pipeline.AddSequentialStage(
		nil,
		ai.NewAgent("prompt_engineer", nil, 3),
		ai.NewAgent("vector_recaller", tools.NewQdrantQuery("caramba", 1536), 3),
		ai.NewAgent("reviewer", &review.Process{}, 3),
		ai.NewAgent("graph_recaller", tools.NewNeo4jQuery(), 3),
		ai.NewAgent("researcher", tools.NewBrowser(), 3),
		ai.NewAgent("vector_memorizer", tools.NewQdrantStore("caramba", 1536), 3),
		ai.NewAgent("graph_memorizer", tools.NewNeo4jStore(), 3),
		ai.NewAgent("developer", tools.NewContainer(), 10),
		ai.NewAgent("vector_memorizer", tools.NewQdrantStore("caramba", 1536), 3),
		ai.NewAgent("graph_memorizer", tools.NewNeo4jStore(), 3),
	)

	return pipeline
}
