package cmd

import (
	"fmt"
	"io"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
)

var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			errnie.SetLevel(log.DebugLevel)

			var (
				wf io.ReadWriter
			)

			switch args[0] {
			case "pipeline":
				errnie.Info("Starting pipeline example")
				wf = examples.NewPipeline()
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}

			evt := event.New(
				"example.pipeline",
				event.MessageEvent,
				event.UserRole,
				message.New(
					message.UserRole,
					"Danny",
					"Hello, how are you?",
				).Marshal(),
			)

			if _, err = io.Copy(wf, evt); err != nil && err != io.EOF {
				return err
			}

			if _, err = io.Copy(cmd.OutOrStdout(), wf); err != nil && err != io.EOF {
				return err
			}

			return nil
		},
	}
)

func init() {
	rootCmd.AddCommand(exampleCmd)
}

var longExample = `
Example demonstrates various capabilities of the Caramba framework.
`
