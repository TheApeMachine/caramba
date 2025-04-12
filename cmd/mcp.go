package cmd

import (
	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/service"
)

var (
	mcpCmd = &cobra.Command{
		Use:   "mcp",
		Short: "Run MCP scenarios",
		Long:  longMCP,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			errnie.SetLevel(log.DebugLevel)

			service := service.NewMCPServer()
			return service.Start()
		},
	}
)

func init() {
	rootCmd.AddCommand(mcpCmd)
}

var longMCP = `
The mcp command starts a Model Context Protocol server.
This allows you to use the Caramba framework with any MCP client.
`
