package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
)

/*
exampleType is a string that represents the type of example to run.
*/
var (
	exampleType string

	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		Run: func(cmd *cobra.Command, args []string) {
			switch exampleType {
			case "research":
				examples.RunResearch()
			case "dev":
				examples.RunDev()
			case "chat":
				examples.RunChat()
			default:
				fmt.Printf("Unknown example type: %s\nAvailable types: research, dev, chat\n", exampleType)
			}
		},
	}
)

/*
init is a function that initializes the example command and sets up the persistent flags.
*/
func init() {
	rootCmd.AddCommand(exampleCmd)
	exampleCmd.Flags().StringVarP(&exampleType, "type", "t", "chat", "Type of example to run (research, dev, chat)")
}

var longExample = `
Run various example scenarios to demonstrate Caramba's capabilities.
Available examples:
  - research: Multi-agent research pipeline
  - dev: Development workflow pipeline
  - chat: Simple chat interaction
`
