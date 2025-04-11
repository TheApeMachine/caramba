package kubrick

import (
	"bufio"
	"context"
	"io"
	"sync"
	"time"

	"github.com/theapemachine/caramba/kubrick/layouts"
	"github.com/theapemachine/caramba/kubrick/types"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
App is a container for a Kubrick application that manages screens and their rendering.
*/
type App struct {
	*types.Contextualizer

	wg           *sync.WaitGroup
	screens      []layouts.Layout
	activeScreen int
	status       types.State
	err          error

	// Rendering infrastructure
	framebuffer *Framebuffer
	transport   Transport
	artifact    *datura.Artifact
	writer      *bufio.Writer
	terminal    *Terminal

	// Synchronization
	mu sync.RWMutex

	// View state
	width  int
	height int
}

type AppOption func(*App)

/*
NewApp creates a new Kubrick application with the specified options.
*/
func NewApp(options ...AppOption) *App {
	artifact := datura.New()

	app := &App{
		Contextualizer: types.NewContextualizer(),
		screens:        make([]layouts.Layout, 0),
		activeScreen:   0,
		status:         types.StateInitialized,
		framebuffer:    NewFramebuffer(),
		transport:      NewStreamTransport(artifact, 1024, 1024),
		artifact:       artifact,
	}

	var err error
	if app.width, app.height, err = app.transport.GetSize(); err != nil {
		errnie.Error(err)
		return nil
	}

	// Ensure the context is set before passing it to screens
	app.Contextualizer.WithContext(context.Background())

	for _, option := range options {
		option(app)
	}

	// Create terminal if we have a screen
	if len(app.screens) > 0 {
		app.terminal = NewTerminal(
			WithRoot(app.screens[app.activeScreen]),
			WithTransport(app.transport),
		)

		app.terminal.WithContext(app.Context())
	}

	app.wg.Add(1)

	if err := app.render(); err != nil {
		errnie.Error(err)
		return nil
	}

	app.status = types.StateRunning
	return app
}

func (app *App) render() error {
	go func() {
		for {
			select {
			case <-app.Done():
				app.Close()
				return
			case <-time.Tick(time.Millisecond * 16):
				if _, app.err = io.Copy(app.transport, app.screens[app.activeScreen]); app.err != nil {
					errnie.Error(app.err)
					app.status = types.StateErrored
					return
				}

				app.writer.Flush()
			}
		}
	}()

	return nil
}

func (app *App) Error() string {
	return app.err.Error()
}

func (app *App) Read(p []byte) (n int, err error) {
	return app.artifact.Read(p)
}

// Write implements io.Writer
func (app *App) Write(p []byte) (n int, err error) {
	if len(app.screens) == 0 {
		return 0, io.EOF
	}

	// Write to active screen
	if n, app.err = app.screens[app.activeScreen].Write(p); app.err != nil {
		app.status = types.StateErrored
		return n, app.err
	}

	return n, nil
}

// Close implements io.Closer
func (app *App) Close() error {
	app.mu.Lock()
	defer app.mu.Unlock()

	app.Cancel()
	app.status = types.StateCanceled

	// Close each screen
	for _, screen := range app.screens {
		if err := screen.Close(); err != nil {
			app.err = err
			return err
		}
	}

	// Clean up resources
	app.framebuffer.Clear()
	app.transport.Close()
	return app.artifact.Close()
}

func WithScreen(screen layouts.Layout) AppOption {
	return func(app *App) {
		app.screens = append(app.screens, screen)

		screen.WithContext(app.Context())

		// Set initial dimensions for the screen
		screen.SetRect(layouts.Rect{
			Pos: layouts.Position{Row: 0, Col: 0},
			Size: layouts.Size{
				Width:  app.width,
				Height: app.height,
			},
		})
	}
}
