package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/cmd"
	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/errnie"
)

func main() {
	// Initialize output with environment settings
	configureOutput()

	// Execute the root command
	if err := cmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// configureOutput sets up the output system based on environment variables
func configureOutput() {
	// Check for output level from environment
	if os.Getenv("CARAMBA_OUTPUT") != "" {
		// Already handled in output package init
	} else if os.Getenv("LOG_LEVEL") == "debug" {
		// If debug logging is enabled, set output to verbose
		output.SetLevel(output.OutputVerbose)
	}

	// Check for color settings
	if os.Getenv("NO_COLOR") != "" || os.Getenv("CARAMBA_NO_COLOR") != "" {
		output.DisableColor()
	}

	// Check for emoji settings
	if os.Getenv("NO_EMOJI") != "" || os.Getenv("CARAMBA_NO_EMOJI") != "" {
		output.DisableEmojis()
	}

	// Configure errnie logger
	configureErrnieLogger()
}

// configureErrnieLogger sets up the errnie logging system
func configureErrnieLogger() {
	// Initialize logger based on environment variables
	if os.Getenv("LOG_LEVEL") == "" {
		os.Setenv("LOG_LEVEL", "info") // Default log level
	}

	if os.Getenv("LOGFILE") == "" {
		// Default to true to keep existing behavior
		os.Setenv("LOGFILE", "true")
	}

	// Initialize the logger
	errnie.InitLogger()

	// Monkeypatch Cobra's error handling to use our error formatting
	cobra.OnInitialize(func() {
		// Register finalization function
		cobra.OnFinalize(func() {
			// Add any custom finalization logic here
		})
	})
}
