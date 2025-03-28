package cmd

import (
	"fmt"

	clog "github.com/charmbracelet/log"
	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"
	"github.com/theapemachine/caramba/pkg/core"
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
			core.NewConfig(core.WithOpenAIAPIKey(openaiAPIKey))

			errnie.SetLevel(clog.DebugLevel)

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
