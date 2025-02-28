package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent/examples"
	"github.com/theapemachine/caramba/pkg/tui"
)

type ErrorUnknownExample struct {
	ExampleType string
}

func (e *ErrorUnknownExample) Error() string {
	return fmt.Sprintf("unknown example type: %s", e.ExampleType)
}

// Example command variables
var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) error {
			switch args[0] {
			case "researcher":
				example := examples.NewResearcher()
				example.Run(cmd.Context())
			default:
				return fmt.Errorf("unknown example type: %s", args[1])
			}

			app := tui.NewApp()
			return app.Start()
		},
	}
)

func init() {
	rootCmd.AddCommand(exampleCmd)
}

var longExample = `
Example demonstrates various capabilities of the Caramba framework.
This command is primarily for testing and demonstration purposes.
`
