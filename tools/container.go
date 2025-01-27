package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/amsh/data"
	"github.com/theapemachine/caramba/tools/container"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type Container struct {
	Reasoning string `json:"reasoning" jsonschema:"title=Reasoning,description=Your reasoning for the next step."`
	Command   string `json:"command" jsonschema:"title=Command,description=The valid bash command to execute for the next step."`
	builder   *container.Builder
	runner    *container.Runner
	Conn      io.ReadWriteCloser
}

func NewContainer() *Container {
	return &Container{
		builder: container.NewBuilder(),
		runner:  container.NewRunner(),
	}
}

func (c *Container) Name() string {
	return "container"
}

func (c *Container) Description() string {
	return "Execute commands in an isolated Debian environment"
}

func (c *Container) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Container]()
}

func (c *Container) Initialize() error {
	if c.Conn == nil {
		if err := os.MkdirAll("/tmp/out", 0755); err != nil {
			return errnie.Error(err)
		}
		if err := os.MkdirAll("/tmp/.ssh", 0755); err != nil {
			return errnie.Error(err)
		}
		if err := os.MkdirAll("/tmp/workspace", 0755); err != nil {
			return errnie.Error(err)
		}

		wd, err := os.Getwd()
		if err != nil {
			return errnie.Error(err)
		}
		
		c.builder.BuildImage(
			context.Background(),
			filepath.Join(wd, "tools", "container", "Dockerfile"),
			container.DefaultImageName,
		)

		conn, err := c.runner.RunContainer(context.Background(), container.DefaultImageName)
		if err != nil {
			return errnie.Error(err)
		}
		c.Conn = conn
	}

	return nil
}

/*
Use the docker container to run the command. This allows the agent to use a fully
featured, isolated Debian environment.
*/
func (c *Container) Use(ctx context.Context, params map[string]any) string {
	if c.Conn == nil {
		if err := c.Initialize(); err != nil {
			return err.Error()
		}
	}

	cmd, ok := params["command"].(string)
	if !ok {
		log.Error("Invalid command parameter received", "params", params)
		return "error: invalid command parameter"
	}

	output := c.runner.ExecuteCommand(ctx, []string{cmd})

	return string(output)
}

func (c *Container) Connect(ctx context.Context, bridge io.ReadWriteCloser) (err error) {
	// Initialize container if needed
	if err := c.Initialize(); err != nil {
		return err
	}

	// Get container connection
	c.Conn, err = c.runner.RunContainer(ctx, container.DefaultImageName)
	if err != nil {
		return err
	}

	return nil
}

func (c *Container) executeCommand(command string, out chan<- *data.Artifact) error {
	// Write command
	if _, err := c.Conn.Write([]byte(command + "\n")); err != nil {
		return fmt.Errorf("failed to write command: %w", err)
	}

	buffer := make([]byte, 4096)
	promptEnd := []byte("# ")

	for {
		n, err := c.Conn.Read(buffer)
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
