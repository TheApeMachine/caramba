package cmd

import (
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
	RunE: func(command *cobra.Command, args []string) error {
		return runProgram(command, "runtime/diffusion.yml", nil)
	},
}

func init() {
	rootCmd.AddCommand(diffusionCmd)
}
