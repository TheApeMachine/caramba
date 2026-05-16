package cmd

import (
	"strings"

	"github.com/spf13/cobra"

	chatpkg "github.com/theapemachine/caramba/pkg/chat"
)

var chatOptions = chatCommandOptions{
	Manifest: chatpkg.DefaultModelManifest,
}

var chatCmd = &cobra.Command{
	Use:          "chat",
	Short:        "Start a terminal chat session.",
	Long:         chatLong,
	SilenceUsage: true,
	RunE:         runChat,
}

func init() {
	rootCmd.AddCommand(chatCmd)

	chatCmd.Flags().StringVar(
		&chatOptions.Manifest,
		"manifest",
		chatOptions.Manifest,
		"embedded or local model manifest",
	)
}

func runChat(command *cobra.Command, _ []string) error {
	generator, err := chatOptions.Generator(command)

	if err != nil {
		return err
	}

	backendName := ""
	modelName := ""

	if namedGenerator, ok := generator.(interface{ BackendName() string }); ok {
		backendName = namedGenerator.BackendName()
	}

	if namedGenerator, ok := generator.(interface{ ModelName() string }); ok {
		modelName = namedGenerator.ModelName()
	}

	session := chatpkg.NewSession(
		command.Context(),
		command.InOrStdin(),
		command.OutOrStdout(),
		generator,
		chatpkg.SessionConfig{
			Runtime:    "model",
			Backend:    backendName,
			Model:      modelName,
			ShowBanner: true,
		},
	)

	return session.Run()
}

type chatCommandOptions struct {
	Manifest string
}

func (options chatCommandOptions) Generator(command *cobra.Command) (chatpkg.Generator, error) {
	return chatpkg.NewModelGenerator(command.Context(), options.ModelConfig())
}

func (options chatCommandOptions) ModelConfig() chatpkg.ModelConfig {
	return chatpkg.ModelConfig{
		Manifest: strings.TrimSpace(options.Manifest),
	}
}

const chatLong = `
Start a terminal chat session.

The chat command is manifest-backed. The manifest declares the runtime, compute
backend, model source, tokenizer source, and generation policy. The CLI only
selects which manifest to run.
`
