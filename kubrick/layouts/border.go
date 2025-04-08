package layouts

import (
	"io"

	"github.com/theapemachine/caramba/kubrick/components"
	"github.com/theapemachine/caramba/pkg/datura"
)

// BorderLayout arranges components in five regions: North, South, East, West, and Center
type BorderLayout struct {
	Spacing    int
	Components []components.Component
	artifact   datura.Artifact
}

func NewBorderLayout(spacing int) *BorderLayout {
	return &BorderLayout{
		Spacing:    spacing,
		Components: make([]components.Component, 0),
		artifact:   datura.New(),
	}
}

const (
	BorderRegionNorth = iota
	BorderRegionSouth
	BorderRegionEast
	BorderRegionWest
	BorderRegionCenter
)

// Read implements io.Reader by reading from all components
func (l *BorderLayout) Read(p []byte) (n int, err error) {
	// Read from each component in sequence
	for _, comp := range l.Components {
		n, err = comp.Read(p)
		if err != io.EOF {
			return n, err
		}
	}

	return 0, io.EOF
}

// Write implements io.Writer by writing to all components
func (l *BorderLayout) Write(p []byte) (n int, err error) {
	// Write to each component
	for _, comp := range l.Components {
		if _, err := comp.Write(p); err != nil {
			return n, err
		}
	}

	return len(p), nil
}

// Close implements io.Closer by closing all components
func (l *BorderLayout) Close() error {
	// Close each component
	for _, comp := range l.Components {
		if err := comp.Close(); err != nil {
			return err
		}
	}

	return nil
}

func WithBorderComponents(components ...components.Component) func(*BorderLayout) {
	return func(l *BorderLayout) {
		l.Components = append(l.Components, components...)
	}
}
