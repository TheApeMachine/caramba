package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/runtime"
	"github.com/theapemachine/errnie"
)

var (
	chatRuntimePath string
)

var chatCmd = &cobra.Command{
	Use:          "chat",
	Short:        "Start a terminal chat session.",
	Long:         chatLong,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) (err error) {
		errnie.Info("Starting chat session...")
		
		session := errnie.Does(func() (*runtime.Session, error) {
			return runtime.OpenSession(cmd.Context(), chatRuntimePath)
		}).Or(func(err error) {
			errnie.Error(err, "Failed to open chat session")
		}).Value()

		if err := session.Run(cmd.Context()); err != nil {
			return fmt.Errorf("chat: %w", err)
		}

		return nil
	},
}

func init() {
	chatCmd.Flags().StringVar(
		&chatRuntimePath,
		"runtime",
		"runtime/chat.yml",
		"Runtime program manifest path under pkg/asset/template/",
	)
	rootCmd.AddCommand(chatCmd)
}

const chatLong = `
Start a terminal chat session.

The chat command runs a runtime program over a model manifest. The model
manifest provides system.topology and weights; the runtime manifest
(default: runtime/chat.yml) declares the decode loop, sampler, and state
objects.
`
