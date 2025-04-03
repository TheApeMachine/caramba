package cmd

import (
	"fmt"
	"io"
	"os"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/stream"
)

var (
	exampleCmd = &cobra.Command{
		Use:   "example [type]",
		Short: "Run example scenarios",
		Long:  longExample,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			core.NewConfig(core.WithOpenAIAPIKey(openaiAPIKey))

			var wf stream.Generator

			switch args[0] {
			case "code":
				wf = examples.NewCode()
			default:
				return fmt.Errorf("unknown example type: %s\nAvailable types: code", args[0])
			}

			streamer := core.NewStreamer(
				core.WithGenerator(wf),
			)

			if _, err = io.Copy(os.Stdout, streamer); err != nil && err != io.EOF {
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

Available examples:
  - pipeline: Demonstrates basic pipeline functionality
  - chat: Shows chat capabilities
  - code: Demonstrates code manipulation
  - memory: Shows memory integration
  - multiagent: Demonstrates multi-agent interaction
  - capnp: Shows Cap'n Proto message serialization
`
