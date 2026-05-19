package cmd

import (
	"github.com/spf13/cobra"
)

var imageCmd = &cobra.Command{
	Use:          "image [prompt]",
	Short:        "Generate an image from a diffusion manifest.",
	Long:         imageLong,
	Args:         cobra.ArbitraryArgs,
	SilenceUsage: true,
	RunE:         func(cmd *cobra.Command, args []string) error { return nil },
}

func init() {
	rootCmd.AddCommand(imageCmd)
}

const imageLong = `
Generate an image from a diffusion manifest.

The model manifest declares the runtime, backend, and Hub asset sources for
each component (tokenizer, text encoder, transformer denoiser, VAE). The
runtime manifest (default: runtime/diffusion.yml) declares the prompt-encode →
denoise-loop → VAE-decode program. The CLI supplies the prompt and can
override the output path.
`
