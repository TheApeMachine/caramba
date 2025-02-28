package tools

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
	"github.com/theapemachine/caramba/pkg/output"
)

// DockerTool provides functionality for running Docker containers and executing commands within them.
type DockerTool struct {
	// client is the Docker API client
	client *client.Client
	// defaultImage is the default Docker image to use
	defaultImage string
	// activeContainers keeps track of running containers
	activeContainers map[string]string // map[containerName]containerId
}

// NewDockerTool creates a new DockerTool with an initialized Docker client.
func NewDockerTool() (*DockerTool, error) {
	output.Info("Creating Docker tool")

	// Initialize Docker client with default options
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	// Ping the Docker daemon to verify connection
	if _, err := cli.Ping(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to connect to Docker daemon: %w", err)
	}

	return &DockerTool{
		client:           cli,
		defaultImage:     "debian:stable-slim",
		activeContainers: make(map[string]string),
	}, nil
}

// Name returns the name of the tool
func (t *DockerTool) Name() string {
	return "docker"
}

// Description returns the description of the tool
func (t *DockerTool) Description() string {
	return "Executes commands in a Docker container environment"
}

// Execute executes the tool with the given arguments
func (t *DockerTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, errors.New("action must be a string")
	}

	switch action {
	case "create":
		return t.createContainer(ctx, args)
	case "execute":
		return t.executeCommand(ctx, args)
	case "list":
		return t.listContainers(ctx)
	case "remove":
		return t.removeContainer(ctx, args)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

// Schema returns the JSON schema for the tool's arguments
func (t *DockerTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"create", "execute", "list", "remove"},
				"description": "Action to perform (create, execute, list, or remove container)",
			},
			"name": map[string]interface{}{
				"type":        "string",
				"description": "Name for the container (required for create, execute, and remove)",
			},
			"image": map[string]interface{}{
				"type":        "string",
				"description": "Docker image to use (defaults to debian:stable-slim)",
			},
			"command": map[string]interface{}{
				"type":        "string",
				"description": "Command to execute in the container (required for execute)",
			},
		},
		"required": []string{"action"},
	}
}

// createContainer creates a new container
func (t *DockerTool) createContainer(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, errors.New("name must be a non-empty string")
	}

	// Use custom image if provided, otherwise use default
	img := t.defaultImage
	if customImage, ok := args["image"].(string); ok && customImage != "" {
		img = customImage
	}

	// Check if image exists locally, if not pull it
	_, _, err := t.client.ImageInspectWithRaw(ctx, img)
	if err != nil {
		output.Info(fmt.Sprintf("Pulling image %s", img))
		reader, err := t.client.ImagePull(ctx, img, image.PullOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to pull image %s: %w", img, err)
		}
		defer reader.Close()

		// Read the output to complete the pull operation
		io.Copy(io.Discard, reader)
	}

	// Create host config with a tmp mount
	hostConfig := &container.HostConfig{
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: "/tmp",
				Target: "/tmp/host",
			},
		},
	}

	// Create the container
	resp, err := t.client.ContainerCreate(
		ctx,
		&container.Config{
			Image:      img,
			Cmd:        []string{"/bin/sh"},
			Tty:        true,
			OpenStdin:  true,
			StdinOnce:  false,
			WorkingDir: "/tmp",
		},
		hostConfig,
		nil,
		nil,
		name,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	// Start the container
	if err := t.client.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	// Store the container ID
	t.activeContainers[name] = resp.ID

	output.Info(fmt.Sprintf("Created container: %s (ID: %s)", name, resp.ID))

	return map[string]interface{}{
		"status":         "success",
		"container_name": name,
		"container_id":   resp.ID,
		"image":          img,
		"message":        fmt.Sprintf("Container '%s' created successfully", name),
	}, nil
}

// executeCommand executes a command in an existing container
func (t *DockerTool) executeCommand(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, errors.New("name must be a non-empty string")
	}

	command, ok := args["command"].(string)
	if !ok || command == "" {
		return nil, errors.New("command must be a non-empty string")
	}

	// Check if container exists
	containerID, exists := t.activeContainers[name]
	if !exists {
		return nil, fmt.Errorf("container with name '%s' does not exist", name)
	}

	// Join command parts into a single string for shell execution
	fullCmd := []string{"/bin/sh", "-c", command}

	// Create exec configuration
	execConfig := container.ExecOptions{
		Cmd:          fullCmd,
		AttachStdout: true,
		AttachStderr: true,
	}

	// Create exec instance in container
	execID, err := t.client.ContainerExecCreate(ctx, containerID, execConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create exec instance: %w", err)
	}

	// Start exec instance
	resp, err := t.client.ContainerExecAttach(ctx, execID.ID, container.ExecStartOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to attach to exec instance: %w", err)
	}
	defer resp.Close()

	// Read stdout and stderr
	var stdout, stderr strings.Builder
	_, err = stdcopy.StdCopy(&stdout, &stderr, resp.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read command output: %w", err)
	}

	// Get exec command exit code
	inspectResp, err := t.client.ContainerExecInspect(ctx, execID.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect exec instance: %w", err)
	}

	output.Info(fmt.Sprintf("Executed command in container %s: %s (exit code: %d)", name, command, inspectResp.ExitCode))

	return map[string]interface{}{
		"status":         "success",
		"container_name": name,
		"container_id":   containerID,
		"command":        command,
		"exit_code":      inspectResp.ExitCode,
		"stdout":         stdout.String(),
		"stderr":         stderr.String(),
	}, nil
}

// listContainers returns a list of all active containers
func (t *DockerTool) listContainers(ctx context.Context) (interface{}, error) {
	containers, err := t.client.ContainerList(ctx, container.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %w", err)
	}

	containerList := []map[string]string{}
	for _, c := range containers {
		// Only include containers managed by this tool
		for name, id := range t.activeContainers {
			if id == c.ID {
				containerList = append(containerList, map[string]string{
					"name":    name,
					"id":      c.ID,
					"image":   c.Image,
					"status":  c.Status,
					"created": fmt.Sprintf("%d", c.Created),
				})
			}
		}
	}

	return map[string]interface{}{
		"status":     "success",
		"containers": containerList,
	}, nil
}

// removeContainer stops and removes a container
func (t *DockerTool) removeContainer(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, errors.New("name must be a non-empty string")
	}

	// Check if container exists
	containerID, exists := t.activeContainers[name]
	if !exists {
		return nil, fmt.Errorf("container with name '%s' does not exist", name)
	}

	// Stop the container
	timeout := 10 // seconds
	if err := t.client.ContainerStop(ctx, containerID, container.StopOptions{Timeout: &timeout}); err != nil {
		return nil, fmt.Errorf("failed to stop container: %w", err)
	}

	// Remove the container
	if err := t.client.ContainerRemove(ctx, containerID, container.RemoveOptions{}); err != nil {
		return nil, fmt.Errorf("failed to remove container: %w", err)
	}

	// Remove from our map
	delete(t.activeContainers, name)

	output.Info(fmt.Sprintf("Removed container: %s", name))

	return map[string]interface{}{
		"status":         "success",
		"container_name": name,
		"container_id":   containerID,
		"message":        fmt.Sprintf("Container '%s' removed successfully", name),
	}, nil
}
