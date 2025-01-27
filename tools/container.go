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
		// Create required directories
		dirs := []string{"/tmp/out", "/tmp/.ssh", "/tmp/workspace"}
		for _, dir := range dirs {
			if err := os.MkdirAll(dir, 0755); err != nil {
				log.Error("Error creating directory", "dir", dir, "error", err)
				return err
			}
		}

		wd, err := os.Getwd()
		if err != nil {
			log.Error("Error getting working directory", "error", err)
			return err
		}

		// Build the image first and handle any errors
		dockerfilePath := filepath.Join(wd, "tools", "container", "Dockerfile")
		if err := c.builder.BuildImage(
			context.Background(),
			dockerfilePath,
			container.DefaultImageName,
		); err != nil {
			log.Error("Error building container image", "error", err)
			return err
		}
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
			log.Error("Error initializing container", "error", err)
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

func (c *Container) Start() error {
	return c.runner.StartContainer(context.Background(), c.runner.GetContainerID())
}

func (c *Container) Connect(ctx context.Context, bridge io.ReadWriteCloser) (err error) {
	// Initialize container if needed
	if err := c.Initialize(); err != nil {
		log.Error("Error initializing container", "error", err)
		return err
	}

	// Get container connection
	c.Conn, err = c.runner.RunContainer(ctx, container.DefaultImageName)
	if err != nil {
		log.Error("Error running container", "error", err)
		return err
	}

	return nil
}

func (c *Container) ExecuteCommand(command string, out chan<- *data.Artifact) error {
	// Write command
	if _, err := c.Conn.Write([]byte(command + "\n")); err != nil {
		log.Error("Error writing command", "error", err)
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

			log.Error("Error reading response", "error", err)
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
