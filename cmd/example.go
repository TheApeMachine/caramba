package cmd

import (
	"fmt"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/core"
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
			core.NewConfig(core.WithOpenAIAPIKey(openaiAPIKey))

			var wf Example

			switch args[0] {
			case "pipeline":
				wf = examples.NewPipeline()
			case "chat":
				wf = examples.NewChat()
			case "code":
				wf = examples.NewCode()
			case "memory":
				wf = examples.NewMemory()
			case "multiagent":
				wf = examples.NewMultiAgent()
			case "capnp":
				wf = examples.NewCapnp()
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}

			return wf.Run()
		},
	}
)

func init() {
	fmt.Println("cmd.example.init")
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
