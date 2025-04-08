package layouts

import (
	"context"
	"time"

	"github.com/theapemachine/caramba/kubrick/components"
	"github.com/theapemachine/caramba/kubrick/types"
	"github.com/theapemachine/caramba/pkg/datura"
)

// ContextAware is implemented by components that can receive a parent context
type ContextAware interface {
	WithContext(context.Context)
}

// GridLayout arranges components in a grid pattern
type GridLayout struct {
	pctx       context.Context
	ctx        context.Context
	cancel     context.CancelFunc
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
	ctx, cancel := context.WithCancel(context.Background())

	layout := &GridLayout{
		ctx:        ctx,
		cancel:     cancel,
		artifact:   datura.New(),
		components: make([]components.Component, 0),
		Rows:       1,
		Columns:    1,
		Spacing:    0,
		status:     types.StateInitialized,
	}

	for _, option := range options {
		option(layout)
	}

	// Only start render loop if we have a parent context
	if layout.pctx != nil {
		go layout.render()
	}

	return layout
}

// renderLoop handles continuous updates from components
func (layout *GridLayout) render() {
	layout.status = types.StateRunning

	for {
		select {
		case <-layout.pctx.Done():
			// A parent component cancelled us
			layout.status = types.StateCanceled
			layout.Close()
			return
		case <-layout.ctx.Done():
			// We cancelled ourselves
			layout.status = types.StateCanceled
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

// WithContext implements ContextAware
func (layout *GridLayout) WithContext(ctx context.Context) {
	layout.pctx = ctx
	// Start render loop now that we have a parent context
	if layout.status == types.StateInitialized {
		go layout.render()
		layout.status = types.StateRunning
	}
}

// SetRect sets the layout's dimensions
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
	layout.cancel() // Cancel our context first
	layout.status = types.StateCanceled
	return layout.artifact.Close()
}

func WithComponents(components ...components.Component) GridLayoutOption {
	return func(l *GridLayout) {
		l.components = components

		// Propagate context to any components that can accept it
		for _, comp := range components {
			if ctxAware, ok := comp.(ContextAware); ok {
				ctxAware.WithContext(l.ctx)
			}
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

func WithContext(ctx context.Context) GridLayoutOption {
	return func(l *GridLayout) {
		l.pctx = ctx
	}
}
