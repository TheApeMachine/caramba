package core

import (
	"io"
	"slices"
)

type Component interface {
	io.ReadWriteCloser
	Optionable
}

type Optionable interface {
	WithOption(option string, value any)
}

// ComponentType defines the role of a component in the system
type ComponentType string

const (
	TypeAgent    ComponentType = "agent"
	TypeTool     ComponentType = "tool"
	TypeContext  ComponentType = "context"
	TypeProcess  ComponentType = "process"
	TypeProvider ComponentType = "provider"
	TypeWorkflow ComponentType = "workflow"
)

// Status represents the current state of a component
type Status string

const (
	StatusIdle     Status = "idle"
	StatusRunning  Status = "running"
	StatusThinking Status = "thinking"
	StatusError    Status = "error"
	StatusSuccess  Status = "success"
)

// BaseComponent provides a standard implementation of Component that other components can embed
type BaseComponent struct {
	IDField            string         `json:"id"`
	ComponentTypeField ComponentType  `json:"component_type"`
	StatusField        Status         `json:"status"`
	OptionsField       map[string]any `json:"options"`

	// Private fields not encoded/decoded by gob
	connections []Component
	input       chan []byte
	output      chan []byte
}

// NewBaseComponent creates a new BaseComponent with the given ID and type
func NewBaseComponent(id string, componentType ComponentType) *BaseComponent {
	return &BaseComponent{
		IDField:            id,
		ComponentTypeField: componentType,
		StatusField:        StatusIdle,
		OptionsField:       make(map[string]any),
		connections:        make([]Component, 0),
		input:              make(chan []byte, 10),
		output:             make(chan []byte, 10),
	}
}

// WithOption implements Optionable
func (c *BaseComponent) WithOption(option string, value any) {
	c.OptionsField[option] = value
}

// ID returns the component's unique identifier
func (c *BaseComponent) ID() string {
	return c.IDField
}

// Type returns the component's type
func (c *BaseComponent) Type() ComponentType {
	return c.ComponentTypeField
}

// Status returns the component's current status
func (c *BaseComponent) Status() Status {
	return c.StatusField
}

// SetStatus updates the component's status
func (c *BaseComponent) SetStatus(status Status) {
	c.StatusField = status
}

// Option returns the value of an option by key
func (c *BaseComponent) Option(key string) (any, bool) {
	value, found := c.OptionsField[key]
	return value, found
}

// Connect establishes a bidirectional connection with another component
func (c *BaseComponent) Connect(other Component) error {
	// Add to our connections if not already present
	found := slices.Contains(c.connections, other)

	if !found {
		c.connections = append(c.connections, other)
	}

	return nil
}

// Disconnect removes a connection with another component
func (c *BaseComponent) Disconnect(other Component) error {
	for i, conn := range c.connections {
		if conn == other {
			// Remove this connection
			c.connections = append(c.connections[:i], c.connections[i+1:]...)
			break
		}
	}

	return nil
}

// Components returns all connected components
func (c *BaseComponent) Components() []Component {
	return c.connections
}

// Read implements io.Reader, reading from input channel
func (c *BaseComponent) Read(p []byte) (n int, err error) {
	// Blocking read from the input channel
	data, ok := <-c.input
	if !ok {
		return 0, io.EOF
	}

	// Copy to the provided buffer
	n = copy(p, data)
	return n, nil
}

// Write implements io.Writer, sending data to output channel
func (c *BaseComponent) Write(p []byte) (n int, err error) {
	// Make a copy of the data
	data := make([]byte, len(p))
	copy(data, p)

	// Send data to output channel
	c.output <- data

	// Distribute data to all connected components
	for _, conn := range c.connections {
		if wc, ok := conn.(io.Writer); ok {
			wc.Write(data)
		}
	}

	return len(p), nil
}

// Close implements io.Closer
func (c *BaseComponent) Close() error {
	close(c.input)
	close(c.output)

	// Disconnect from all components
	for _, conn := range c.connections {
		c.Disconnect(conn)
	}

	return nil
}
