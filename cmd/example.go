package cmd

import (
	"fmt"
	"io"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			errnie.SetLevel(log.DebugLevel)

			var (
				wf  io.ReadWriteCloser
				msg *core.Message
			)

			switch args[0] {
			case "pipeline":
				errnie.Info("Starting pipeline example")
				wf = examples.NewPipeline()
				msg = core.NewMessage("user", "Danny", "Hello, how are you?")
			case "discussion":
				errnie.Info("Starting discussion example")
				wf = examples.NewDiscussion()
				msg = core.NewMessage("user", "Danny", "Discuss AI Agents")
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}

			if _, err = io.Copy(wf, core.NewEvent(msg, nil)); err != nil && err != io.EOF {
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
