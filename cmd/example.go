package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/errnie"
)

/*
exampleType is a string that represents the type of example to run.
*/
var (
	exampleType string

	exampleCmd = &cobra.Command{
		Use:   "example [name]",
		Short: "Run an example",
		Long:  longExample,
		Run: func(cmd *cobra.Command, args []string) {
			os.Setenv("LOG_LEVEL", "debug")
			os.Setenv("LOGFILE", "true")

			errnie.InitLogger()

			if len(args) == 0 {
				fmt.Println("Please specify an example name")
				return
			}

			switch args[0] {
			case "coding-agent":
				examples.NewExampleAgent(examples.NewCodingAgent()).Run()
			default:
				fmt.Printf("Unknown example: %s\n", args[0])
			}
		},
	}
)

/*
init is a function that initializes the example command and sets up the persistent flags.
*/
func init() {
	rootCmd.AddCommand(exampleCmd)
	exampleCmd.Flags().StringVarP(&exampleType, "type", "t", "chat", "Type of example to run")
}

var longExample = `
Run various example scenarios to demonstrate Caramba's capabilities.

Available examples:

  - browser-agent: Demonstrates basic web browsing and information extraction
  - research-agent: Shows in-depth research capabilities across multiple sources
  - interactive-agent: Demonstrates complex web interactions and analysis
  - coding-agent: Demonstrates coding capabilities with a simple Go program
`
