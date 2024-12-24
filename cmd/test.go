package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/process/chainofthought"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Used to test the agent",
	Long:  `Executes a test setup`,
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// in := make(chan string)
		// consumer := utils.NewConsumer()
		// go consumer.Print(in)

		for event := range ai.NewPipeline(
			"Are influencers responsible for the spread of influenza?",
		).AddSequentialStage(
			ai.NewAgent("reasoner", &chainofthought.Process{}, 3),
			ai.NewAgent("researcher", tools.NewBrowser(), 5),
		).Execute() {
			fmt.Print(event.Content)
		}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
