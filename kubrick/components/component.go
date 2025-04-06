package components

import (
	"bufio"
	"io"
)

// Position represents a 2D coordinate in the terminal
type Position struct {
	Row int
	Col int
}

// Size represents dimensions in the terminal
type Size struct {
	Width  int
	Height int
}

// Rect represents a rectangular region in the terminal
type Rect struct {
	Pos  Position
	Size Size
}

// Component is the base interface that all UI components must implement
type Component interface {
	io.ReadWriteCloser

	// Render draws the component to the provided writer
	Render(writer *bufio.Writer) error

	// GetRect returns the component's current position and size
	GetRect() Rect

	// SetRect updates the component's position and size
	SetRect(rect Rect)

	// IsDirty returns true if the component needs to be redrawn
	IsDirty() bool

	// SetDirty marks the component as needing a redraw
	SetDirty(dirty bool)
}

// Container is a component that can hold other components
type Container interface {
	Component

	// AddComponent adds a child component
	AddComponent(component Component)

	// RemoveComponent removes a child component
	RemoveComponent(component Component)

	// GetComponents returns all child components
	GetComponents() []Component
}

// Layout is an interface for objects that can arrange components
type Layout interface {
	// Arrange positions all components within the container
	Arrange(container Container)
}

// BaseComponent provides common functionality for components
type BaseComponent struct {
	rect  Rect
	dirty bool

	// Buffered I/O
	buffer []byte
	pos    int
}

func (b *BaseComponent) GetRect() Rect {
	return b.rect
}

func (b *BaseComponent) SetRect(rect Rect) {
	if b.rect != rect {
		b.rect = rect
		b.dirty = true
	}
}

func (b *BaseComponent) IsDirty() bool {
	return b.dirty
}

func (b *BaseComponent) SetDirty(dirty bool) {
	b.dirty = dirty
}

// Read implements io.Reader
func (b *BaseComponent) Read(p []byte) (n int, err error) {
	if b.pos >= len(b.buffer) {
		return 0, io.EOF
	}
	n = copy(p, b.buffer[b.pos:])
	b.pos += n
	return n, nil
}

// Write implements io.Writer
func (b *BaseComponent) Write(p []byte) (n int, err error) {
	b.buffer = append(b.buffer, p...)
	b.dirty = true
	return len(p), nil
}

// Close implements io.Closer
func (b *BaseComponent) Close() error {
	b.buffer = nil
	b.pos = 0
	return nil
}

// BaseContainer provides common container functionality
type BaseContainer struct {
	BaseComponent
	components []Component
	layout     Layout
}

func (c *BaseContainer) AddComponent(component Component) {
	c.components = append(c.components, component)
	c.dirty = true
}

func (c *BaseContainer) RemoveComponent(component Component) {
	for i, comp := range c.components {
		if comp == component {
			c.components = append(c.components[:i], c.components[i+1:]...)
			c.dirty = true
			break
		}
	}
}

func (c *BaseContainer) GetComponents() []Component {
	return c.components
}

func (c *BaseContainer) SetLayout(layout Layout) {
	c.layout = layout
	c.dirty = true
}

func (c *BaseContainer) Render(writer *bufio.Writer) error {
	if c.layout != nil {
		c.layout.Arrange(c)
	}

	for _, component := range c.components {
		if err := component.Render(writer); err != nil {
			return err
		}
	}

	c.dirty = false
	return nil
}

// Read implements io.Reader by reading from all child components
func (c *BaseContainer) Read(p []byte) (n int, err error) {
	// First read our own buffer
	n, err = c.BaseComponent.Read(p)
	if err != io.EOF {
		return n, err
	}

	// Then try reading from child components
	for _, comp := range c.components {
		n, err = comp.Read(p)
		if err != io.EOF {
			return n, err
		}
	}

	return 0, io.EOF
}

// Write implements io.Writer by writing to all child components
func (c *BaseContainer) Write(p []byte) (n int, err error) {
	// Write to our own buffer
	n, err = c.BaseComponent.Write(p)
	if err != nil {
		return n, err
	}

	// Write to all child components
	for _, comp := range c.components {
		if _, err := comp.Write(p); err != nil {
			return n, err
		}
	}

	return n, nil
}

// Close implements io.Closer by closing all child components
func (c *BaseContainer) Close() error {
	// Close our own resources
	if err := c.BaseComponent.Close(); err != nil {
		return err
	}

	// Close all child components
	for _, comp := range c.components {
		if err := comp.Close(); err != nil {
			return err
		}
	}

	return nil
}
