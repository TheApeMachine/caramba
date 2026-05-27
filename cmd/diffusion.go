package cmd

import (
	"io"
	"strings"

	"github.com/spf13/cobra"
)

var diffusionCmd = &cobra.Command{
	Use:   "diffusion",
	Short: "Run the diffusion image-generation runtime program.",
	Long: "Loads runtime/diffusion.yml, resolves the included HF model components " +
		"(transformer / text_encoder / vae), and runs one denoising loop to " +
		"produce an image. The default include set is FLUX.2 Klein 4B; swap " +
		"the include sources in the YAML to target a different FLUX-2-shaped " +
		"checkpoint without recompiling.",
	Args: cobra.ArbitraryArgs,
	RunE: func(command *cobra.Command, args []string) error {
		return runProgramWithInput(command, "runtime/diffusion.yml", nil, diffusionPromptReader(args))
	},
}

func diffusionPromptReader(args []string) io.Reader {
	if len(args) == 0 {
		return nil
	}

	return strings.NewReader(strings.Join(args, " ") + "\n")
}

func init() {
	rootCmd.AddCommand(diffusionCmd)
}
