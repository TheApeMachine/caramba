package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

/*
testCmd is a command that is used to test the agent.
*/
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Used to test the agent",
	Long:  `Executes a test setup`,
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		config := agent.NewConfig(
			"default",
			"coordinator",
			utils.NewName(),
			tools.NewToolset(),
		)

		generator := agent.NewGenerator(
			config,
			provider.NewBalancedProvider(),
		)

		executor := agent.NewExecutor(
			config,
			generator,
		)

		stream.NewConsumer().Print(executor.Generate(provider.NewMessage(
			provider.RoleUser,
			"How many times do we find the letter r in the word strawberry?",
		)), false)
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
