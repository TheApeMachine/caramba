package tui

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/kubrick"
	"github.com/theapemachine/caramba/kubrick/components/spinner"
	"github.com/theapemachine/caramba/kubrick/layouts"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Simple struct {
	app      *kubrick.App
	artifact *datura.ArtifactBuilder
}

func NewSimple() *Simple {
	return &Simple{
		app: kubrick.NewApp(
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
		),
		artifact: datura.New(),
	}
}

func (simple *Simple) Run() (err error) {
	for err == nil {
		// Create a new artifact for each iteration
		simple.artifact = datura.New()

		if _, err = io.Copy(os.Stdout, simple); err != nil {
			if err == io.EOF {
				return nil
			}
			errnie.Error(err)
		}

		// Cleanup the artifact after each iteration
		if err := simple.artifact.Close(); err != nil {
			errnie.Error(err)
		}
	}

	return nil
}

func (simple *Simple) Read(p []byte) (n int, err error) {
	// Read directly from the app to the output buffer
	if n, err = simple.app.Read(p); err != nil {
		if err == io.EOF {
			return 0, io.EOF
		}
		return n, err
	}

	return n, nil
}

func (simple *Simple) Write(p []byte) (n int, err error) {
	return simple.app.Write(p)
}

func (simple *Simple) Close() error {
	simple.app.Close()
	return simple.artifact.Close()
}
