package cmd

import (
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/service"
	"github.com/theapemachine/caramba/pkg/stores/inmemory"
	"github.com/theapemachine/caramba/pkg/task"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
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
				service.WithTaskManager(task.NewManager(
					task.WithTaskStore(inmemory.NewRepository()),
					task.WithLLMProvider(provider.NewOpenAIProvider(
						provider.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
					)),
				)),
			).Listen("3210")
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

	rootCmd.AddCommand(serveCmd)
}

var longServe = `
Serve a caramba A2A agent.
`
