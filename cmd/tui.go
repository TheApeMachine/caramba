package cmd

import (
	"io"
	"os"
	"sync"
	"time"

	"github.com/spf13/cobra"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/theapemachine/caramba/kubrick"
	"github.com/theapemachine/caramba/kubrick/components/spinner"
	"github.com/theapemachine/caramba/kubrick/layouts"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	tuiCmd = &cobra.Command{
		Use:   "tui",
		Short: "Run Caramba TUI",
		Long:  longTui,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			errnie.SetOutput(false)

			var wg sync.WaitGroup

			wg.Add(1)
			app := kubrick.NewApp(
				kubrick.WithScreen(
					layouts.NewGridLayout(
						layouts.WithRows(1),
						layouts.WithColumns(1),
						layouts.WithSpacing(1),
						layouts.WithComponents(
							spinner.NewSpinner(),
						),
					),
				),
			)

			app.WithContext(cmd.Context())

			tick := time.Second / 60

			go func() {
				defer wg.Done()

				for {
					select {
					case <-cmd.Context().Done():
						return
					case <-time.Tick(tick):
						io.Copy(os.Stdout, app)
					}
				}
			}()

			wg.Wait()
			return app
		},
	}
)

func init() {
	rootCmd.AddCommand(tuiCmd)
}

var longTui = `
Serve a caramba TUI.
`
