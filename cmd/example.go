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
		Use:   "example [type]",
		Short: "Run an example",
		Long: `Run one of the available examples:
  - research: Run the research pipeline example
  - dev: Run the development pipeline example
  - chat: Run the simple chat example
  - pipeline: Run the graph-based pipeline example`,
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) == 0 {
				fmt.Println("Please specify an example type")
				return
			}

			switch args[0] {
			case "research":
				examples.RunResearch()
			case "dev":
				examples.RunDev()
			case "chat":
				examples.RunChat()
			case "pipeline":
				examples.RunPipeline()
			default:
				fmt.Printf("Unknown example type: %s\n", args[0])
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
