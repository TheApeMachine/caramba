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
	Runtime:           "auto",
	Backend:           "auto",
	RepoType:          "model",
	Manifest:          chatpkg.DefaultModelManifest,
	MaxNewTokens:      128,
	RepetitionPenalty: 1.1,
	Temperature:       0.8,
	TopK:              50,
	TopP:              0.95,
	StopSpecialTokens: true,
	StreamDelay:       12 * time.Millisecond,
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

	chatCmd.Flags().StringVar(&chatOptions.Runtime, "runtime", chatOptions.Runtime, "chat runtime: auto, preview, or model")
	chatCmd.Flags().StringVar(&chatOptions.Backend, "backend", chatOptions.Backend, "model compute backend: auto, cpu, metal, cuda, or xla")
	chatCmd.Flags().StringVar(&chatOptions.Model, "model", "", "model repo, hf locator, or local model directory")
	chatCmd.Flags().StringVar(&chatOptions.Manifest, "manifest", chatOptions.Manifest, "embedded or local model manifest")
	chatCmd.Flags().StringVar(&chatOptions.Tokenizer, "tokenizer", "", "tokenizer repo, hf locator, or local tokenizer directory")
	chatCmd.Flags().StringVar(&chatOptions.Revision, "revision", "", "Hub revision for model and tokenizer assets")
	chatCmd.Flags().StringVar(&chatOptions.RepoType, "repo-type", chatOptions.RepoType, "Hub repo type: model, dataset, or space")
	chatCmd.Flags().StringVar(&chatOptions.Cache, "cache", "", "Hub cache directory")
	chatCmd.Flags().StringVar(&chatOptions.Prompt, "prompt", "", "run one prompt and exit")
	chatCmd.Flags().IntVar(&chatOptions.MaxNewTokens, "max-new-tokens", chatOptions.MaxNewTokens, "maximum generated tokens for model runtime")
	chatCmd.Flags().Float64Var(&chatOptions.RepetitionPenalty, "repetition-penalty", chatOptions.RepetitionPenalty, "deterministic penalty applied to tokens already in the context; 1 disables it")
	chatCmd.Flags().Float64Var(&chatOptions.Temperature, "temperature", chatOptions.Temperature, "model sampling temperature; 0 uses greedy selection")
	chatCmd.Flags().IntVar(&chatOptions.TopK, "top-k", chatOptions.TopK, "keep only the top K logits before sampling; 0 disables top-k")
	chatCmd.Flags().Float64Var(&chatOptions.TopP, "top-p", chatOptions.TopP, "keep the smallest probability mass >= top-p before sampling")
	chatCmd.Flags().Int64Var(&chatOptions.Seed, "seed", chatOptions.Seed, "model sampling seed")
	chatCmd.Flags().StringArrayVar(&chatOptions.StopTokens, "stop-token", chatOptions.StopTokens, "text token or special token that stops generation; repeatable")
	chatCmd.Flags().BoolVar(&chatOptions.StopSpecialTokens, "stop-special-tokens", chatOptions.StopSpecialTokens, "stop generation when the tokenizer emits any special token")
	chatCmd.Flags().DurationVar(&chatOptions.StreamDelay, "stream-delay", chatOptions.StreamDelay, "preview stream delay per text segment")
}

func runChat(command *cobra.Command, _ []string) error {
	generator, err := chatOptions.Generator(command)

	if err != nil {
		return err
	}

	runtimeName := chatOptions.RuntimeName(command)
	backendName := ""

	if runtimeName == "model" {
		backendName = chatOptions.Backend

		if namedGenerator, ok := generator.(interface{ BackendName() string }); ok {
			backendName = namedGenerator.BackendName()
		}
	}

	session := chatpkg.NewSession(
		command.Context(),
		command.InOrStdin(),
		command.OutOrStdout(),
		generator,
		chatpkg.SessionConfig{
			Runtime:    runtimeName,
			Backend:    backendName,
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
	Runtime           string
	Backend           string
	Model             string
	Manifest          string
	Tokenizer         string
	Revision          string
	RepoType          string
	Cache             string
	Prompt            string
	MaxNewTokens      int
	RepetitionPenalty float64
	Temperature       float64
	TopK              int
	TopP              float64
	Seed              int64
	StopTokens        []string
	StopSpecialTokens bool
	StreamDelay       time.Duration
}

func (options chatCommandOptions) Generator(command *cobra.Command) (chatpkg.Generator, error) {
	switch options.RuntimeName(command) {
	case "preview":
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
				Backend:           options.Backend,
				Model:             options.Model,
				Tokenizer:         options.Tokenizer,
				Manifest:          options.Manifest,
				Cache:             options.Cache,
				Revision:          options.Revision,
				RepoType:          options.RepoType,
				MaxNewTokens:      options.MaxNewTokens,
				RepetitionPenalty: options.RepetitionPenalty,
				Temperature:       options.Temperature,
				TopK:              options.TopK,
				TopP:              options.TopP,
				Seed:              options.Seed,
				StopTokens:        options.StopTokens,
				StopSpecialTokens: options.StopSpecialTokens,
			},
		)
	default:
		return nil, fmt.Errorf("chat: unsupported runtime %q", options.Runtime)
	}
}

func (options chatCommandOptions) RuntimeName(command *cobra.Command) string {
	runtime := strings.TrimSpace(options.Runtime)

	if runtime == "" || runtime == "auto" {
		if command.Flags().Changed("model") || command.Flags().Changed("manifest") {
			return "model"
		}

		return "preview"
	}

	return runtime
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

The auto runtime uses the manifest-backed model runtime when --model or
--manifest is provided. Without a model, it starts the tokenizer preview shell.
`
