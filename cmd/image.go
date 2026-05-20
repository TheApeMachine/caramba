package cmd

import (
	"context"
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/runtime"
)

var (
	imageRuntimePath string
	imageOutputPath  string
)

var imageCmd = &cobra.Command{
	Use:          "image [prompt]",
	Short:        "Generate an image from a diffusion manifest.",
	Long:         imageLong,
	Args:         cobra.ArbitraryArgs,
	SilenceUsage: true,
	RunE:         runImage,
}

func init() {
	imageCmd.Flags().StringVar(
		&imageRuntimePath,
		"runtime",
		"runtime/diffusion.yml",
		"Runtime program manifest path under pkg/asset/template/",
	)
	imageCmd.Flags().StringVar(
		&imageOutputPath,
		"output",
		"",
		"Override output image path",
	)
	rootCmd.AddCommand(imageCmd)
}

func runImage(command *cobra.Command, args []string) error {
	_ = command

	prompt := ""

	if len(args) > 0 {
		prompt = args[0]
	}

	if prompt == "" {
		return fmt.Errorf("image: prompt is required")
	}

	ctx := context.Background()
	session, err := runtime.OpenSession(ctx, imageRuntimePath)

	if err != nil {
		return fmt.Errorf("image: %w", err)
	}

	initial := map[string]any{
		"prompt": prompt,
	}

	if imageOutputPath != "" {
		initial["generation.output"] = imageOutputPath
	}

	if err := session.RunWithValues(ctx, initial); err != nil {
		return fmt.Errorf("image: %w", err)
	}

	return nil
}

const imageLong = `
Generate an image from a diffusion manifest.

The model manifest declares the runtime, backend, and Hub asset sources for
each component (tokenizer, text encoder, transformer denoiser, VAE). The
runtime manifest (default: runtime/diffusion.yml) declares the prompt-encode →
denoise-loop → VAE-decode program. The CLI supplies the prompt and can
override the output path.
`
