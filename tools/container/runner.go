package container

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/charmbracelet/log"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
)

const ContainerName = "caramba-terminal"

/*
Runner encapsulates the functionality for running and interacting with Docker containers.
It provides methods to create, start, and attach to containers, allowing for seamless
integration with external systems such as language models.
*/
type Runner struct {
	client      *client.Client
	containerID string
}

/*
NewRunner initializes a new Runner instance with a Docker client.
This setup allows for interaction with the Docker daemon on the host system.
*/
func NewRunner() *Runner {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil
	}
	return &Runner{client: cli}
}

/*
Attach to the running container.
*/
func (r *Runner) Attach(ctx context.Context, containerID string) (io.ReadWriteCloser, error) {
	// Attach to the container
	attachResp, err := r.client.ContainerAttach(ctx, containerID, container.AttachOptions{
		Stream: true,
		Stdin:  true,
		Stdout: true,
		Stderr: true,
		Logs:   true,
	})

	if err != nil {
		return nil, err
	}

	r.containerID = containerID
	return attachResp.Conn, nil
}

/*
RunContainer creates, starts, and attaches to a new container based on the specified image.
It provides channels for stdin and stdout/stderr, enabling interactive communication with the container.
This method is particularly useful for integrating with language models or other interactive processes.

Parameters:
  - ctx: The context for the Docker API calls
  - imageName: The name of the Docker image to use
  - cmd: The command to run in the container
  - username: The username to create and use within the container
  - customMessage: A message to be displayed when attaching to the container

Returns:
  - in: A channel for sending input to the container
  - out: A channel for receiving output from the container
  - err: Any error encountered during the process
*/
func (r *Runner) RunContainer(ctx context.Context, imageName string) (io.ReadWriteCloser, error) {
	// Check if container already exists
	containers, err := r.client.ContainerList(ctx, container.ListOptions{All: true})
	if err != nil {
		return nil, err
	}

	var existingContainer string
	for _, c := range containers {
		for _, name := range c.Names {
			if name == "/"+ContainerName {
				existingContainer = c.ID
				break
			}
		}
	}

	// If container exists and is running, reuse it
	if existingContainer != "" {
		// Check if container is running
		inspect, err := r.client.ContainerInspect(ctx, existingContainer)
		if err != nil {
			return nil, err
		}

		if inspect.State.Running {
			r.containerID = existingContainer
			// Reattach to the existing container
			return r.Attach(ctx, existingContainer)
		}

		// If container exists but not running, remove it
		if err := r.client.ContainerRemove(ctx, existingContainer, container.RemoveOptions{Force: true}); err != nil {
			return nil, err
		}
	}

	// Create host config with volume mount
	hostConfig := &container.HostConfig{
		Mounts: []mount.Mount{
			{
				Type:   mount.TypeBind,
				Source: "/tmp/workspace",
				Target: "/tmp/out",
			},
			{
				Type:   mount.TypeBind,
				Source: "/tmp/.ssh",
				Target: "/home/user/.ssh",
			},
		},
	}

	// Create the container with specific configuration
	resp, err := r.client.ContainerCreate(ctx, &container.Config{
		Image:     DefaultImageName,
		Cmd:       []string{"/bin/bash"},
		Tty:       true,
		OpenStdin: true,
		StdinOnce: false,
		Env: []string{
			fmt.Sprintf("USERNAME=%s", "user"),
			"TERM=xterm",
			"PS1=\\u@\\h:\\w\\$ ",
		},
		WorkingDir: "/tmp/workspace",
	}, hostConfig, nil, nil, ContainerName)

	if err != nil {
		return nil, err
	}

	if err = r.client.ContainerStart(
		context.Background(), resp.ID, container.StartOptions{},
	); err != nil {
		return nil, err
	}

	statusCh, errCh := r.client.ContainerWait(
		context.Background(),
		resp.ID,
		container.WaitConditionNotRunning,
	)

	select {
	case err = <-errCh:
		if err != nil {
			return nil, err
		}
	case <-statusCh:
	}

	return r.Attach(ctx, resp.ID)
}

/*
StopContainer gracefully stops a running container.
This method ensures proper cleanup of container resources.

Parameters:
  - ctx: The context for the Docker API calls
  - containerID: The ID of the container to stop
*/
func (r *Runner) StopContainer(ctx context.Context) error {
	timeout := 10 // seconds
	return r.client.ContainerStop(ctx, r.containerID, container.StopOptions{Timeout: &timeout})
}

/*
ExecuteCommand executes a command in the container and returns the output.
*/
func (r *Runner) ExecuteCommand(ctx context.Context, cmd []string) []byte {
	commandStr := strings.Join(cmd, " ")
	fullCmd := []string{"/bin/bash", "-c", commandStr}

	execConfig := container.ExecOptions{
		Cmd:          fullCmd,
		AttachStdout: true,
		AttachStderr: true,
	}

	execIDResp, err := r.client.ContainerExecCreate(ctx, r.containerID, execConfig)
	if err != nil {
		return nil
	}

	execAttachResp, err := r.client.ContainerExecAttach(ctx, execIDResp.ID, container.ExecStartOptions{})
	if err != nil {
		return nil
	}
	defer execAttachResp.Close()

	var stdout, stderr strings.Builder
	_, err = stdcopy.StdCopy(&stdout, &stderr, execAttachResp.Reader)
	if err != nil {
		log.Error(err)
		return nil
	}

	return []byte(stdout.String() + stderr.String())
}
