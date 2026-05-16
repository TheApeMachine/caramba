package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/theapemachine/caramba/pkg/diffusion"
)

var imageOptions = imageCommandOptions{
	Manifest: diffusion.DefaultManifest,
}

var imageCmd = &cobra.Command{
	Use:          "image [prompt]",
	Short:        "Generate an image from a diffusion manifest.",
	Long:         imageLong,
	Args:         cobra.ArbitraryArgs,
	SilenceUsage: true,
	RunE:         runImage,
}

func init() {
	rootCmd.AddCommand(imageCmd)

	imageCmd.Flags().StringVar(
		&imageOptions.Manifest,
		"manifest",
		imageOptions.Manifest,
		"embedded or local diffusion manifest",
	)
	imageCmd.Flags().StringVar(
		&imageOptions.Prompt,
		"prompt",
		"",
		"text prompt to encode",
	)
	imageCmd.Flags().StringVar(
		&imageOptions.Output,
		"output",
		"",
		"output PNG path; default comes from the manifest",
	)
}

func runImage(command *cobra.Command, args []string) error {
	options := imageOptions

	if len(args) > 0 && strings.TrimSpace(options.Prompt) == "" {
		options.Prompt = strings.Join(args, " ")
	}

	if strings.TrimSpace(options.Prompt) == "" {
		return fmt.Errorf("image: prompt is required")
	}

	var result diffusion.Result

	err := runWithQPoolProgress(command, func() error {
		pipeline, err := diffusion.NewPipeline(command.Context(), options.Config())

		if err != nil {
			return err
		}

		defer pipeline.Close()

		result, err = pipeline.Generate(command.Context(), options.Prompt)

		return err
	})

	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(
		command.OutOrStdout(),
		"caramba image output=%s size=%dx%d\n",
		result.Output,
		result.Width,
		result.Height,
	)

	return err
}

type imageCommandOptions struct {
	Manifest string
	Prompt   string
	Output   string
}

func (options imageCommandOptions) Config() diffusion.Config {
	return diffusion.Config{
		Manifest: strings.TrimSpace(options.Manifest),
		Prompt:   strings.TrimSpace(options.Prompt),
		Output:   strings.TrimSpace(options.Output),
	}
}

const imageLong = `
Generate an image from a diffusion manifest.

The image command is manifest-backed. The manifest declares the runtime,
backend, Hub assets, tokenizer, text encoder, transformer denoiser, scheduler,
and generation defaults. The CLI supplies the prompt and can override only the
output path.
`
