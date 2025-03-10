package cmd

import (
	"bytes"
	"fmt"
	"io"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
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

			buf := new(bytes.Buffer)
			if _, err = io.Copy(buf, wf); err != nil {
				return err
			}

			wf.Close()

			fmt.Println(buf.String())

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
