package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/diffusion"
)

var (
	diagnoseFullMode      bool
	diagnoseNormTraceMode bool
)

var diffusionDiagnoseCmd = &cobra.Command{
	Use:   "diffusion-diagnose [prompt]",
	Short: "Lightweight FLUX conditioning checks (3 denoiser forwards by default).",
	Long: `Encodes the prompt, then runs a small number of denoiser forwards to test timestep
and text conditioning. Does not load the VAE or run the full denoise loop unless --full.

Prefer: make build && ./caramba diffusion-diagnose "prompt"
Avoid repeated go run — it recompiles and spikes memory.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(command *cobra.Command, args []string) error {
		prompt := "An elephant playing chess"

		if len(args) > 0 {
			prompt = args[0]
		}

		options := diffusion.DefaultOptions()

		if diagnoseFullMode {
			options.IncludeNormTrace = true
		}

		if diagnoseNormTraceMode {
			options.IncludeNormTrace = true
		}

		fmt.Fprintf(
			os.Stderr,
			"diffusion-diagnose: text encoder + %s denoiser forwards (use built binary, not go run)...\n",
			forwardBudgetLabel(options),
		)

		return diffusion.RunDiagnosticsWithOptions(command.Context(), prompt, os.Stdout, options)
	},
}

func forwardBudgetLabel(options diffusion.Options) string {
	if options.IncludeNormTrace {
		return "7+"
	}

	return "3"
}

func init() {
	diffusionDiagnoseCmd.Flags().BoolVar(
		&diagnoseFullMode,
		"full",
		false,
		"Run legacy heavy diagnostics (~9 denoiser forwards + host latent trace).",
	)
	diffusionDiagnoseCmd.Flags().BoolVar(
		&diagnoseNormTraceMode,
		"norm-trace",
		false,
		"Add four-forward latent norm trace (large host↔device transfers).",
	)

	rootCmd.AddCommand(diffusionDiagnoseCmd)
}
