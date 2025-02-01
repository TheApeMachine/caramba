/*
Copyright © 2025 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"context"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/tools"
)

// containerCmd represents the container command
var containerCmd = &cobra.Command{
	Use:   "container",
	Short: "A brief description of your command",
	Long:  `Test container`,
	Run: func(cmd *cobra.Command, args []string) {
		instance := tools.NewContainer()
		instance.Initialize(context.Background())
		instance.RunCommandInteractive(context.Background(), "echo hello")
	},
}

func init() {
	rootCmd.AddCommand(containerCmd)
}
