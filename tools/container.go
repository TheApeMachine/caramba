package tools

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/tools/container"
	"github.com/theapemachine/caramba/utils"
)

type Container struct {
	Reasoning string `json:"reasoning" jsonschema:"title=Reasoning,description=Your reasoning for the next step."`
	Command   string `json:"command" jsonschema:"title=Command,description=The valid bash command to execute for the next step."`
	builder   *container.Builder
	runner    *container.Runner
	reader    io.ReadCloser
	writer    io.WriteCloser
}

func NewContainer() *Container {
	builder := container.NewBuilder()
	runner, err := container.NewRunner()
	if err != nil {
		log.Error("Error creating runner", "error", err)
		return nil
	}
	return &Container{builder: builder, runner: runner}
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

// Initialize builds the image and runs the container if the connection is not yet established.
func (c *Container) Initialize(ctx context.Context) error {
	if c.reader == nil {
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
		if err := c.builder.BuildImage(ctx, dockerfilePath, container.DefaultImageName); err != nil {
			log.Error("Error building container image", "error", err)
			return err
		}

		// Run the container (we attach to the container's TTY)
		connIn, connOut, err := c.runner.RunContainer(ctx, container.DefaultImageName, []string{"/bin/bash"}, "user", "Terminal ready")
		if err != nil {
			log.Error("Error running container", "error", err)
			return err
		}

		// For simplicity, we assume a single connection serves both reading and writing.
		// (Docker returns the same Conn for in/out in our setup.)
		c.writer = connIn
		c.reader = connOut
	}
	return nil
}

// Connect ensures that the container connection is set up.
func (c *Container) Connect(ctx context.Context) error {
	return c.Initialize(ctx)
}

// RunCommandInteractive executes a command in the container and returns the complete output.
func (c *Container) RunCommandInteractive(ctx context.Context, cmd string) (string, error) {
	if c.runner == nil {
		if err := c.Connect(ctx); err != nil {
			return "", err
		}
	}

	// Always ensure the command ends with a newline
	if !strings.HasSuffix(cmd, "\n") {
		cmd += "\n"
	}

	// Execute the command using Docker's exec API
	output, err := c.runner.ExecuteCommand(ctx, []string{cmd})

	// Return the raw output and error as is
	return string(output), err
}
