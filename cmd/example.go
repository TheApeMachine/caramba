package cmd

import (
	"fmt"
	"os"

	"github.com/charmbracelet/log"
	"github.com/containerd/containerd/v2/cmd/containerd/command"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Example interface {
	Run() (err error)
}

var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			errnie.SetLevel(log.DebugLevel)

			// Start the containerd daemon, so the environment tool can use it.
			go func() {
				// We pass in the command context so that the containerd daemon is
				// shutdown when the command finishes for any reason.
				if err := command.App().RunContext(
					cmd.Context(),
					os.Args,
				); errnie.Error(err) != nil {
					os.Exit(1)
				}
			}()

			var (
				wf Example
			)

			switch args[0] {
			case "pipeline":
				wf = examples.NewPipeline()
			case "chat":
				wf = examples.NewChat()
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}

			return wf.Run()
		},
	}
)

func init() {
	rootCmd.AddCommand(exampleCmd)
}

var longExample = `
Example demonstrates various capabilities of the Caramba framework.
`
