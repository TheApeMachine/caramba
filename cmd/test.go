package cmd

import (
	"os"

	"github.com/spf13/cobra"
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
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
