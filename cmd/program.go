package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/config"
	carambaruntime "github.com/theapemachine/caramba/pkg/runtime"
)

var programCmd = &cobra.Command{
	Use:   "program [path]",
	Short: "Run one manifest program",
	Args:  cobra.MaximumNArgs(1),
	RunE:  runProgram,
}

func init() {
	rootCmd.AddCommand(programCmd)
}

func runProgram(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()

	if ctx == nil {
		ctx = context.Background()
	}

	manifestPath := programPath

	if len(args) > 0 {
		manifestPath = args[0]
	}

	if manifestPath == "" {
		return fmt.Errorf("program: manifest path is required")
	}

	computeConfig := config.NewComputeConfig()
	hubConfig := config.NewHubConfig()
	platform, err := carambaruntime.NewPlatform(ctx, carambaruntime.PlatformOptions{
		ComputeConfig: computeConfig,
		HubConfig:     hubConfig,
		Stdin:         os.Stdin,
	})

	if err != nil {
		return err
	}

	defer platform.Close()

	fmt.Fprintf(os.Stderr, "caramba program: device=%s program=%s\n", computeConfig.Device, manifestPath)

	return platform.RunProgram(ctx, manifestPath, carambaruntime.PlatformOptions{
		ComputeConfig: computeConfig,
		HubConfig:     hubConfig,
		Stdin:         os.Stdin,
	})
}
