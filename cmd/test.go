package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/system"
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

		pool := system.NewPool()
		name := utils.NewName()
		pool.Add("default", "coordinator", name, tools.NewToolset(
			&tools.Team{},
		))

		stream.NewConsumer().Print(pool.Select(name).Generate(provider.NewMessage(
			provider.RoleUser,
			"How many times do we find the letter r in the word strawberry?",
		)), false)
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
