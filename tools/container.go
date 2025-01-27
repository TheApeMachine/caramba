package tools

import (
	"context"
	"io"
	"os"
	"path/filepath"

	"github.com/charmbracelet/log"
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

		_, _, err = c.runner.RunContainer(context.Background(), container.DefaultImageName, []string{"/bin/bash"}, "user", "Terminal ready")
		if err != nil {
			log.Error("Error running container", "error", err)
			return err
		}
	}

	return nil
}

func (c *Container) Connect(ctx context.Context, bridge io.ReadWriteCloser) (err error) {
	return nil
}

/*
Use the docker container to run the command. This allows the agent to use a fully
featured, isolated Debian environment.
*/
func (c *Container) Use(ctx context.Context, params map[string]any) string {
	if c.Conn == nil {
		if err := c.Connect(context.Background(), nil); err != nil {
			log.Error("Error connecting to container", "error", err)
			return err.Error()
		}
	}

	cmd, ok := params["command"].(string)
	if !ok {
		log.Error("Invalid command parameter received", "params", params)
		return "error: invalid command parameter"
	}

	output, err := c.runner.ExecuteCommand(ctx, []string{cmd})
	if err != nil {
		log.Error("Error executing command", "error", err)
		return err.Error()
	}

	return string(output)
}

func (c *Container) ExecuteCommand(cmd string) string {
	output, err := c.runner.ExecuteCommand(context.Background(), []string{cmd})
	if err != nil {
		log.Error("Error executing command", "error", err)
		return err.Error()
	}
	return string(output)
}
