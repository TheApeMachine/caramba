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
	conn      io.ReadWriteCloser
}

func NewContainer() *Container {
	return &Container{
		builder: container.NewBuilder(),
		runner:  container.NewRunner(),
	}
}

func (c *Container) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Container]()
}

func (c *Container) Initialize() error {
	if c.conn == nil {
		if err := os.MkdirAll("/tmp/out", 0755); err != nil {
			return errnie.Error(err)
		}
		if err := os.MkdirAll("/tmp/.ssh", 0755); err != nil {
			return errnie.Error(err)
		}

		wd, err := os.Getwd()
		if err != nil {
			return errnie.Error(err)
		}
		c.builder.BuildImage(
			context.Background(),
			filepath.Join(wd, "tools", "container", "Dockerfile"),
			"caramba-dev",
		)

		conn, err := c.runner.RunContainer(context.Background(), "caramba-dev")
		if err != nil {
			return errnie.Error(err)
		}
		c.conn = conn
	}

	return nil
}

/*
Use the docker container to run the command. This allows the agent to use a fully
featured, isolated Debian environment.
*/
func (c *Container) Use(ctx context.Context, params map[string]any) string {
	if c.conn == nil {
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

func (c *Container) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
	c.conn = conn
	containerConn, err := c.runner.RunContainer(ctx, "caramba-dev")
	if err != nil {
		return err
	}
	c.conn = containerConn

	return nil
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
