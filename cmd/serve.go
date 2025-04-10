package cmd

import (
	"fmt"
	"sync"

	"github.com/spf13/cobra"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service"
	"github.com/theapemachine/caramba/pkg/twoface"
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
			serviceType := args[0]

			var wg sync.WaitGroup
			wg.Add(1)

			switch serviceType {
			case "hub":
				hub, err := twoface.NewHub(cmd.Context(), &wg)

				if err != nil {
					return errnie.InternalError(err)
				}

				hub.Start()
			case "agent":
				if _, err = service.NewAgentComponent(
					cmd.Context(),
					&wg,
					name,
					initialMessage,
					tools,
					subscriptions,
				); err != nil {
					return errnie.InternalError(err)
				}
			case "tool":
				if _, err = service.NewToolComponent(
					cmd.Context(),
					&wg,
					name,
					subscriptions,
				); err != nil {
					return errnie.InternalError(err)
				}
			case "provider":
				if _, err = service.NewProviderComponent(
					cmd.Context(),
					&wg,
					name,
					subscriptions,
				); err != nil {
					return errnie.InternalError(err)
				}
			default:
				return fmt.Errorf("unknown service type: %s", serviceType)
			}

			wg.Wait()

			return nil
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
