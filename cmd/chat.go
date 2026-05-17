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
		"embedded or local model manifest (provides system.topology + weights)",
	)
	chatCmd.Flags().StringVar(
		&chatOptions.RuntimeManifest,
		"runtime",
		chatOptions.RuntimeManifest,
		"embedded or local runtime manifest (defaults to runtime/chat.yml)",
	)
	chatCmd.Flags().StringVar(
		&chatOptions.ProvenanceOutput,
		"provenance",
		chatOptions.ProvenanceOutput,
		"path to write the run's provenance ledger",
	)
}

func runChat(command *cobra.Command, _ []string) error {
	return runWithQPoolProgress(command, func() error {
		generator, err := chatpkg.NewRuntimeModelGenerator(
			command.Context(), chatOptions.ModelConfig(),
		)

		if err != nil {
			return err
		}

		session := chatpkg.NewSession(
			command.Context(),
			command.InOrStdin(),
			command.OutOrStdout(),
			generator,
			chatpkg.SessionConfig{
				Runtime:    "model",
				Backend:    generator.BackendName(),
				Model:      generator.ModelName(),
				ShowBanner: true,
			},
		)

		if err := session.Run(); err != nil {
			return err
		}

		if chatOptions.ProvenanceOutput != "" {
			return generator.WriteLedger(chatOptions.ProvenanceOutput)
		}

		return nil
	})
}

type chatCommandOptions struct {
	Manifest         string
	RuntimeManifest  string
	ProvenanceOutput string
}

func (options chatCommandOptions) ModelConfig() chatpkg.ModelConfig {
	return chatpkg.ModelConfig{
		Manifest:        strings.TrimSpace(options.Manifest),
		RuntimeManifest: strings.TrimSpace(options.RuntimeManifest),
	}
}

const chatLong = `
Start a terminal chat session.

The chat command runs a runtime program over a model manifest. The model
manifest provides system.topology and weights; the runtime manifest
(default: runtime/chat.yml) declares the decode loop, sampler, and state
objects.
`
