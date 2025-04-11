package cmd

import (
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/service"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
)

var (
	name           string
	tools          []string
	subscriptions  []string
	initialMessage string

	serveCmd = &cobra.Command{
		Use:   "serve [hub|agent|tool]",
		Short: "Run Caramba services",
		Long:  longServe,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			return service.NewMCP().Start()
		},
	}
)

func init() {
	serveCmd.Flags().StringVar(
		&name,
		"name",
		"",
		"Unique name for the agent or tool service instance",
	)

	serveCmd.Flags().StringSliceVar(
		&tools,
		"tools",
		[]string{},
		"Tools for the agent or tool service instance",
	)

	serveCmd.Flags().StringSliceVar(
		&subscriptions,
		"subscriptions",
		[]string{},
		"Subscriptions for the agent or tool service instance",
	)

	serveCmd.Flags().StringVar(
		&initialMessage,
		"initial-message",
		"",
		"Initial message for the agent or tool service instance",
	)
	rootCmd.AddCommand(serveCmd)
}

var longServe = `
Serve a caramba component.

Available services:
  - hub     : Serve a caramba Hub
  - agent   : Serve a caramba agent
  - tool    : Serve a caramba tool
  - provider: Serve a caramba provider
`
