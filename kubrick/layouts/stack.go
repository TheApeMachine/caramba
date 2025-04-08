package layouts

import (
	"io"

	"github.com/theapemachine/caramba/kubrick/components"
	"github.com/theapemachine/caramba/pkg/datura"
)

// StackLayout arranges components vertically or horizontally
type StackLayout struct {
	Vertical   bool
	Spacing    int
	Components []components.Component
	artifact   datura.Artifact
}

func NewVerticalStackLayout(spacing int) *StackLayout {
	return &StackLayout{
		Vertical:   true,
		Spacing:    spacing,
		Components: make([]components.Component, 0),
		artifact:   datura.New(),
	}
}

func NewHorizontalStackLayout(spacing int) *StackLayout {
	return &StackLayout{
		Vertical:   false,
		Spacing:    spacing,
		Components: make([]components.Component, 0),
		artifact:   datura.New(),
	}
}

// Read implements io.Reader by reading from all components
func (layout *StackLayout) Read(p []byte) (n int, err error) {
	// Read from each component in sequence
	for _, comp := range layout.Components {
		n, err = comp.Read(p)
		if err != io.EOF {
			return n, err
		}
	}

	return 0, io.EOF
}

// Write implements io.Writer by writing to all components
func (layout *StackLayout) Write(p []byte) (n int, err error) {
	// Write to each component
	for _, comp := range layout.Components {
		if _, err := comp.Write(p); err != nil {
			return n, err
		}
	}

	return len(p), nil
}

// Close implements io.Closer by closing all components
func (layout *StackLayout) Close() error {
	// Close each component
	for _, comp := range layout.Components {
		if err := comp.Close(); err != nil {
			return err
		}
	}

	return nil
}

func WithStackComponents(components ...components.Component) func(*StackLayout) {
	return func(layout *StackLayout) {
		layout.Components = append(layout.Components, components...)
	}
}
