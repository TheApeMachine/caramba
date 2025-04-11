package layouts

import (
	"context"
	"sync"
	"time"

	"github.com/theapemachine/caramba/kubrick/components"
	"github.com/theapemachine/caramba/kubrick/types"
	"github.com/theapemachine/caramba/pkg/datura"
)

type GridLayout struct {
	*types.Contextualizer

	wg         *sync.WaitGroup
	artifact   *datura.Artifact
	components []components.Component
	Rows       int
	Columns    int
	Spacing    int
	rect       Rect
	status     types.State
	err        error
}

type GridLayoutOption func(*GridLayout)

func NewGridLayout(options ...GridLayoutOption) *GridLayout {
	layout := &GridLayout{
		Contextualizer: types.NewContextualizer(),
		wg:             &sync.WaitGroup{},
		artifact:       datura.New(),
		components:     make([]components.Component, 0),
		Rows:           1,
		Columns:        1,
		Spacing:        0,
		status:         types.StateInitialized,
	}

	// Ensure the context is set before passing it to components
	layout.Contextualizer.WithContext(context.Background())

	for _, option := range options {
		option(layout)
	}

	layout.wg.Add(1)

	if err := layout.render(); err != nil {
		layout.err = err
	}

	return layout
}

// renderLoop handles continuous updates from components
func (layout *GridLayout) render() (err error) {
	layout.status = types.StateRunning

	go func() {
		layout.wg.Wait()

		for {
			select {
			case <-layout.Done():
				layout.Close()
				return
			case <-time.After(10 * time.Millisecond):
				if layout.status != types.StateRunning {
					return
				}

				// Update component positions based on current rect
				layout.updatePositions()

				// Read from all components and combine their artifacts
				for _, comp := range layout.components {
					buf := make([]byte, 1024)
					if n, err := comp.Read(buf); err == nil && n > 0 {
						layout.artifact.Write(buf[:n])
					}
				}
			}
		}
	}()

	return nil
}

// updatePositions calculates and sets the position of each component
func (layout *GridLayout) updatePositions() {
	// Calculate cell dimensions
	cellWidth := (layout.rect.Size.Width - (layout.Spacing * (layout.Columns - 1))) / layout.Columns
	cellHeight := (layout.rect.Size.Height - (layout.Spacing * (layout.Rows - 1))) / layout.Rows

	// Position each component
	for i, comp := range layout.components {
		if i >= layout.Rows*layout.Columns {
			break
		}

		row := i / layout.Columns
		col := i % layout.Columns

		// Calculate component position
		x := layout.rect.Pos.Col + (col * (cellWidth + layout.Spacing))
		y := layout.rect.Pos.Row + (row * (cellHeight + layout.Spacing))

		if container, ok := comp.(Layout); ok {
			container.SetRect(Rect{
				Pos: Position{
					Row: y,
					Col: x,
				},
				Size: Size{
					Width:  cellWidth,
					Height: cellHeight,
				},
			})
		}
	}
}

func (layout *GridLayout) SetRect(rect Rect) {
	layout.rect = rect
}

// Read implements io.Reader - streams the rendered view
func (layout *GridLayout) Read(p []byte) (n int, err error) {
	if n, layout.err = layout.artifact.Read(p); layout.err != nil {
		layout.status = types.StateErrored
		return n, layout.err
	}
	return n, layout.err
}

// Write implements io.Writer - updates layout state based on commands
func (layout *GridLayout) Write(p []byte) (n int, err error) {
	if n, layout.err = layout.artifact.Write(p); layout.err != nil {
		layout.status = types.StateErrored
		return n, layout.err
	}
	return n, layout.err
}

// Close implements io.Closer
func (layout *GridLayout) Close() error {
	layout.Cancel()
	layout.status = types.StateCanceled
	return layout.artifact.Close()
}

func (layout *GridLayout) WithContext(ctx context.Context) {
	layout.Contextualizer.WithContext(ctx)
}

func WithComponents(components ...components.Component) GridLayoutOption {
	return func(l *GridLayout) {
		l.components = components

		for _, comp := range components {
			comp.WithContext(l.Context())
		}
	}
}

func WithRows(rows int) GridLayoutOption {
	return func(l *GridLayout) {
		l.Rows = rows
	}
}

func WithColumns(columns int) GridLayoutOption {
	return func(l *GridLayout) {
		l.Columns = columns
	}
}

func WithSpacing(spacing int) GridLayoutOption {
	return func(l *GridLayout) {
		l.Spacing = spacing
	}
}
