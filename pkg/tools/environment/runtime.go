package environment

import (
	"context"
	"io"
)

/*
Runtime defines the interface for container runtime operations.

It provides a common set of methods for managing container lifecycle,
executing commands, and handling container I/O operations. This interface
allows for different container runtime implementations (e.g., Docker, containerd)
while maintaining a consistent API.
*/
type Runtime interface {
	// CreateContainer initializes a new container instance in the runtime.
	// Returns an error if container creation fails.
	CreateContainer(ctx context.Context) error

	// StartContainer starts a previously created container.
	// Returns an error if the container fails to start.
	StartContainer(ctx context.Context) error

	// StopContainer stops a running container.
	// Returns an error if the container fails to stop.
	StopContainer(ctx context.Context) error

	// AttachIO connects I/O streams to a container for interactive operations.
	// The stdin reader is used for input, while stdout and stderr writers receive output.
	// Returns an error if stream attachment fails.
	AttachIO(stdin io.Reader, stdout, stderr io.Writer) error

	// ExecuteCommand runs a command inside the container.
	// Command output is written to the provided stdout and stderr writers.
	// Returns an error if command execution fails.
	ExecuteCommand(ctx context.Context, command string, stdout, stderr io.Writer) error

	// PullImage downloads a container image from a registry.
	// The ref parameter specifies the image to pull.
	// Returns an error if the image pull fails.
	PullImage(ctx context.Context, ref string) error

	// BuildImage creates a new container image from a Dockerfile.
	// The dockerfile parameter contains the Dockerfile contents, and tag specifies the image name.
	// Returns an error if image building fails.
	BuildImage(ctx context.Context, dockerfile []byte, tag string) error
}
