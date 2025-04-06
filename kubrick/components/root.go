package components

import (
	"bufio"
)

// RootContainer is the top-level container that manages all components
type RootContainer struct {
	BaseContainer
	width  int
	height int
}

func NewRootContainer() *RootContainer {
	return &RootContainer{}
}

// UpdateSize updates the container dimensions and marks it dirty if changed
func (r *RootContainer) UpdateSize(width, height int) {
	if r.width != width || r.height != height {
		r.width = width
		r.height = height
		r.SetRect(Rect{
			Pos:  Position{0, 0},
			Size: Size{width, height},
		})
		r.SetDirty(true)
	}
}

// RenderTo renders all components to the provided writer
func (r *RootContainer) RenderTo(writer *bufio.Writer) error {
	if !r.IsDirty() {
		return nil
	}

	// Hide cursor during rendering
	if _, err := writer.WriteString("\033[?25l"); err != nil {
		return err
	}

	// Ensure cursor is shown after rendering
	defer writer.WriteString("\033[?25h")

	// Let the base container handle rendering of child components
	if err := r.Render(writer); err != nil {
		return err
	}

	return nil
}

// IsDirty returns true if any component needs redrawing
func (r *RootContainer) IsDirty() bool {
	if r.dirty {
		return true
	}

	// Check if any child component is dirty
	for _, comp := range r.components {
		if comp.IsDirty() {
			return true
		}
	}

	return false
}
