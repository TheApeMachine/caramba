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
			log.Info("Starting example")

			var wf io.ReadWriteCloser

			switch args[0] {
			case "pipeline":
				wf = examples.NewPipeline()
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}

			if _, err = io.Copy(wf, core.NewEvent(
				core.NewMessage("user", "Danny", "Hello, how are you?"),
				nil,
			)); err != nil && err != io.EOF {
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
