package cmd

import (
	"fmt"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/stream"
)

var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			core.NewConfig(core.WithOpenAIAPIKey(openaiAPIKey))

			var wf stream.Generator

			switch args[0] {
			case "code":
				wf = examples.NewCode()
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}

			for artifact := range wf.Generate(make(chan *datura.Artifact)) {
				fmt.Println(artifact.String())
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
