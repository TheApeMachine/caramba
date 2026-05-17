package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	runtimepkg "github.com/theapemachine/caramba/pkg/runtime"
)

var imageOptions = imageCommandOptions{
	Manifest: runtimepkg.DefaultDiffusionManifest,
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
		"embedded or local diffusion manifest (provides component sources + topologies)",
	)
	imageCmd.Flags().StringVar(
		&imageOptions.RuntimeManifest,
		"runtime",
		imageOptions.RuntimeManifest,
		"embedded or local runtime manifest (defaults to runtime/diffusion.yml)",
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
	imageCmd.Flags().StringVar(
		&imageOptions.ProvenanceOutput,
		"provenance",
		"",
		"path to write the run's provenance ledger",
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

	var result runtimepkg.Result

	err := runWithQPoolProgress(command, func() error {
		pipeline, err := runtimepkg.NewRuntimeDiffusionPipeline(command.Context(), options.Config())

		if err != nil {
			return err
		}

		defer pipeline.Close()

		result, err = pipeline.Generate(command.Context(), options.Prompt)

		if err != nil {
			return err
		}

		if options.ProvenanceOutput != "" {
			return pipeline.WriteLedger(options.ProvenanceOutput)
		}

		return nil
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
	Manifest         string
	RuntimeManifest  string
	Prompt           string
	Output           string
	ProvenanceOutput string
}

func (options imageCommandOptions) Config() runtimepkg.DiffusionConfig {
	return runtimepkg.DiffusionConfig{
		Manifest:        strings.TrimSpace(options.Manifest),
		RuntimeManifest: strings.TrimSpace(options.RuntimeManifest),
		Prompt:          strings.TrimSpace(options.Prompt),
		Output:          strings.TrimSpace(options.Output),
	}
}

const imageLong = `
Generate an image from a diffusion manifest.

The model manifest declares the runtime, backend, and Hub asset sources for
each component (tokenizer, text encoder, transformer denoiser, VAE). The
runtime manifest (default: runtime/diffusion.yml) declares the prompt-encode →
denoise-loop → VAE-decode program. The CLI supplies the prompt and can
override the output path.
`
