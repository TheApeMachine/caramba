package cmd

import (
	"fmt"
	"os"
	"time"

	clog "github.com/charmbracelet/log"
	"github.com/containerd/containerd/v2/client"
	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/containerd/containerd/v2/cmd/containerd/command"
	"github.com/containerd/log"
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

			// Set up containerd to use our custom logger
			logger := NewContainerdLogger()
			log.G(cmd.Context())                             // Initialize the global logger
			log.L = logger.WithField("module", "containerd") // Set our logger as the default

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

			// Wait for the containerd daemon to start.
			for {
				conn, err := client.New(
					"/var/run/containerd/containerd.sock",
					client.WithDefaultNamespace("caramba"),
				)

				if errnie.Error(err) != nil {
					time.Sleep(1 * time.Second)
					continue
				}

				conn.Close()
				break
			}

			var (
				wf Example
			)

			switch args[0] {
			case "pipeline":
				wf = examples.NewPipeline()
			case "chat":
				wf = examples.NewChat()
			case "code":
				wf = examples.NewCode()
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
