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
	ctx          context.Context
	cancel       context.CancelFunc
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
	ctx, cancel := context.WithCancel(context.Background())

	artifact := datura.New()

	app := &App{
		ctx:          ctx,
		cancel:       cancel,
		screens:      make([]layouts.Layout, 0),
		activeScreen: 0,
		status:       types.StateInitialized,
		framebuffer:  NewFramebuffer(),
		transport:    NewStreamTransport(artifact, 1024, 1024),
		artifact:     artifact,
	}

	var err error
	if app.width, app.height, err = app.transport.GetSize(); err != nil {
		errnie.Error(err)
		return nil
	}

	for _, option := range options {
		option(app)
	}

	// Create terminal if we have a screen
	if len(app.screens) > 0 {
		app.terminal = NewTerminal(
			WithRoot(app.screens[app.activeScreen]),
			WithTransport(app.transport),
		)
	}

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
			case <-app.ctx.Done():
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

	app.cancel() // Cancel our context first
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

// WithContext implements layouts.ContextAware
func (app *App) WithContext(ctx context.Context) {
	app.mu.Lock()
	defer app.mu.Unlock()
	app.ctx = ctx
}

func WithScreen(screen layouts.Layout) AppOption {
	return func(app *App) {
		// Propagate context if screen supports it
		if ctxAware, ok := screen.(layouts.ContextAware); ok {
			ctxAware.WithContext(app.ctx)
		}
		app.screens = append(app.screens, screen)

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
