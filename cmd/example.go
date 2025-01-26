package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/errnie"
)

/*
exampleType is a string that represents the type of example to run.
*/
var (
	exampleType string

	exampleCmd = &cobra.Command{
		Use:   "example [type]",
		Short: "Run an example",
		Long:  longExample,
		Run: func(cmd *cobra.Command, args []string) {
			os.Setenv("LOG_LEVEL", "debug")
			os.Setenv("LOGFILE", "true")

			errnie.InitLogger()

			if len(args) == 0 {
				fmt.Println("Please specify an example type")
				return
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
