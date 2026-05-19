package cmd

import (
	"github.com/spf13/cobra"
)

var chatCmd = &cobra.Command{
	Use:          "chat",
	Short:        "Start a terminal chat session.",
	Long:         chatLong,
	SilenceUsage: true,
	RunE:         func(cmd *cobra.Command, args []string) error { return nil },
}

func init() {
	rootCmd.AddCommand(chatCmd)
}

const chatLong = `
Start a terminal chat session.

The chat command runs a runtime program over a model manifest. The model
manifest provides system.topology and weights; the runtime manifest
(default: runtime/chat.yml) declares the decode loop, sampler, and state
objects.
`
