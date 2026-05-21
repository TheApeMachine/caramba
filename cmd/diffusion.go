package cmd

import (
	"strings"

	"github.com/spf13/cobra"
)

var diffusionCmd = &cobra.Command{
	Use:   "diffusion [prompt]",
	Short: "Run the diffusion image generation runtime program.",
	Long:  "Loads runtime/diffusion.yml, encodes the prompt, denoises latents, and writes the output image.",
	Args:  cobra.MinimumNArgs(1),
	RunE: func(command *cobra.Command, args []string) error {
		prompt := strings.TrimSpace(strings.Join(args, " "))

		return runProgram(command, "runtime/diffusion.yml", map[string]any{
			"prompt": prompt,
		})
	},
}

func init() {
	rootCmd.AddCommand(diffusionCmd)
}
