package cmd

import (
	"fmt"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
)

type Example interface {
	Run()
}

var (
	exampleCmd = &cobra.Command{
		Use:   "example [type]",
		Short: "Run example scenarios",
		Long:  longExample,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			var wf Example

			switch args[0] {
			case "code":
				wf = examples.NewCode()
			default:
				return fmt.Errorf("unknown example type: %s\nAvailable types: code", args[0])
			}

			wf.Run()

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
