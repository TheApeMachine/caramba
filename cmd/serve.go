package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/service"
)

var (
	name  string
	tools []string

	serveCmd = &cobra.Command{
		Use:   "serve [hub|agent|tool]",
		Short: "Run Caramba services",
		Long:  longServe,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			return service.NewA2A(
				service.WithName(name),
				service.WithAgent(agent.NewBuilder(
					agent.WithTaskManager(agent.NewManager(
						agent.WithLLMProvider(provider.NewOpenAIProvider(
							provider.WithOpenAIAPIKey(os.Getenv("OPENAI_API_KEY")),
						)),
					)),
				)),
			).Listen("3210")
		},
	}
)

func init() {
	os.Setenv("USE_REDIS", "true")

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

	rootCmd.AddCommand(serveCmd)
}

var longServe = `
Serve a caramba A2A agent.
`
