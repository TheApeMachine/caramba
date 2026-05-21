package cmd

import (
	"github.com/spf13/cobra"
)

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Run the interactive chat runtime program.",
	Long:  "Loads runtime/chat.yml, resolves the included HF model, and streams tokens from stdin.",
	RunE: func(command *cobra.Command, args []string) error {
		return runProgram(command, "runtime/chat.yml", nil)
	},
}

func init() {
	rootCmd.AddCommand(chatCmd)
}
