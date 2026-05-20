package cmd

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/runtime"
)

var (
	chatRuntimePath string
)

var chatCmd = &cobra.Command{
	Use:          "chat",
	Short:        "Start a terminal chat session.",
	Long:         chatLong,
	SilenceUsage: true,
	RunE:         runChat,
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

func runChat(command *cobra.Command, args []string) error {
	_ = command
	_ = args

	ctx := context.Background()
	session, err := runtime.OpenSession(ctx, chatRuntimePath)

	if err != nil {
		return fmt.Errorf("chat: %w", err)
	}

	if err := session.Run(ctx); err != nil {
		return fmt.Errorf("chat: %w", err)
	}

	return nil
}

const chatLong = `
Start a terminal chat session.

The chat command runs a runtime program over a model manifest. The model
manifest provides system.topology and weights; the runtime manifest
(default: runtime/chat.yml) declares the decode loop, sampler, and state
objects.
`
