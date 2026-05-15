package cmd

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"

	chatpkg "github.com/theapemachine/caramba/pkg/chat"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

var chatOptions = chatCommandOptions{
	Runtime:      "preview",
	RepoType:     "model",
	MaxNewTokens: 128,
	StreamDelay:  12 * time.Millisecond,
}

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Start a terminal chat session.",
	Long:  chatLong,
	RunE:  runChat,
}

func init() {
	rootCmd.AddCommand(chatCmd)

	chatCmd.Flags().StringVar(&chatOptions.Runtime, "runtime", chatOptions.Runtime, "chat runtime: preview or model")
	chatCmd.Flags().StringVar(&chatOptions.Model, "model", "", "model repo, hf locator, or local model directory")
	chatCmd.Flags().StringVar(&chatOptions.Tokenizer, "tokenizer", "", "tokenizer repo, hf locator, or local tokenizer directory")
	chatCmd.Flags().StringVar(&chatOptions.Revision, "revision", "", "Hub revision for model and tokenizer assets")
	chatCmd.Flags().StringVar(&chatOptions.RepoType, "repo-type", chatOptions.RepoType, "Hub repo type: model, dataset, or space")
	chatCmd.Flags().StringVar(&chatOptions.Cache, "cache", "", "Hub cache directory")
	chatCmd.Flags().StringVar(&chatOptions.Prompt, "prompt", "", "run one prompt and exit")
	chatCmd.Flags().IntVar(&chatOptions.MaxNewTokens, "max-new-tokens", chatOptions.MaxNewTokens, "maximum generated tokens for model runtime")
	chatCmd.Flags().DurationVar(&chatOptions.StreamDelay, "stream-delay", chatOptions.StreamDelay, "preview stream delay per text segment")
}

func runChat(command *cobra.Command, _ []string) error {
	generator, err := chatOptions.Generator(command)

	if err != nil {
		return err
	}

	session := chatpkg.NewSession(
		command.Context(),
		command.InOrStdin(),
		command.OutOrStdout(),
		generator,
		chatpkg.SessionConfig{
			Runtime:    chatOptions.Runtime,
			Model:      chatOptions.Model,
			ShowBanner: chatOptions.Prompt == "",
		},
	)

	if strings.TrimSpace(chatOptions.Prompt) != "" {
		return session.RunPrompt(chatOptions.Prompt)
	}

	return session.Run()
}

type chatCommandOptions struct {
	Runtime      string
	Model        string
	Tokenizer    string
	Revision     string
	RepoType     string
	Cache        string
	Prompt       string
	MaxNewTokens int
	StreamDelay  time.Duration
}

func (options chatCommandOptions) Generator(command *cobra.Command) (chatpkg.Generator, error) {
	switch strings.TrimSpace(options.Runtime) {
	case "", "preview":
		return chatpkg.NewPreviewGenerator(
			command.Context(),
			chatpkg.PreviewConfig{
				TokenizerSource: options.TokenizerSource(),
				StreamDelay:     options.StreamDelay,
			},
		)
	case "model":
		return chatpkg.NewModelGenerator(
			command.Context(),
			chatpkg.ModelConfig{
				Model:           options.Model,
				TokenizerSource: options.TokenizerSource(),
				MaxNewTokens:    options.MaxNewTokens,
			},
		)
	default:
		return nil, fmt.Errorf("chat: unsupported runtime %q", options.Runtime)
	}
}

func (options chatCommandOptions) TokenizerSource() tokenizer.Source {
	source := strings.TrimSpace(options.Tokenizer)

	if source == "" {
		source = strings.TrimSpace(options.Model)
	}

	if source == "" {
		return tokenizer.Source{}
	}

	return tokenizer.Source{
		Source:   source,
		Cache:    options.Cache,
		Revision: options.Revision,
		RepoType: options.RepoType,
	}
}

const chatLong = `
Start a terminal chat session.

The default preview runtime streams through the same terminal path the model
runtime will use, and can optionally load a tokenizer from --model or
--tokenizer. The model runtime is reserved for the CausalLM runner.
`
