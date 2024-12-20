package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/invopop/jsonschema"
	"github.com/theapemachine/amsh/data"
	"github.com/theapemachine/caramba/tools/container"
	"github.com/theapemachine/errnie"
)

type Container struct {
	Name    string
	builder *container.Builder
	runner  *container.Runner
	conn    io.ReadWriteCloser
}

func NewContainer() *Container {
	return &Container{
		builder: container.NewBuilder(),
		runner:  container.NewRunner(),
	}
}

func (c *Container) GenerateSchema() string {
	schema := jsonschema.Reflect(&Container{})
	out, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		errnie.Error(err)
	}
	return string(out)
}

func (c *Container) Use(params map[string]any) string {
	if c.conn == nil {
		return "error: container not connected"
	}

	// Extract command from params
	cmd, ok := params["command"].(string)
	if !ok {
		return "error: invalid command parameter"
	}

	// Execute command in container
	ctx := context.Background()
	output := c.runner.ExecuteCommand(ctx, []string{cmd})
	return string(output)
}

func (c *Container) Connect(conn io.ReadWriteCloser) {
	c.conn = conn
	ctx := context.Background()
	containerConn, err := c.runner.RunContainer(ctx, "caramba-dev")
	if err != nil {
		return
	}
	c.conn = containerConn
}

func (c *Container) executeCommand(command string, out chan<- *data.Artifact) error {
	// Write command
	if _, err := c.conn.Write([]byte(command + "\n")); err != nil {
		return fmt.Errorf("failed to write command: %w", err)
	}

	buffer := make([]byte, 4096)
	promptEnd := []byte("# ")

	for {
		n, err := c.conn.Read(buffer)
		if err != nil {
			if err == io.EOF {
				// Handle EOF: Tool might have finished, but process any remaining data
				if n > 0 {
					chunk := buffer[:n]
					out <- data.New("container", "tool", "interactive", chunk)
				}
				return nil
			}
			return fmt.Errorf("failed to read response: %w", err)
		}

		if n > 0 {
			chunk := buffer[:n]
			out <- data.New("container", "tool", "interactive", chunk)

			if bytes.HasSuffix(chunk, promptEnd) {
				return nil
			}
		}
	}
}
