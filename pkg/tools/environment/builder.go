package environment

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
Builder manages the creation and initialization of container environments.
It handles container runtime setup and provides a buffered interface for
environment operations.
*/
type Builder struct {
	buffer  *stream.Buffer
	Runtime Runtime
}

/*
NewBuilder creates a new Builder instance.

It initializes a container runtime (defaulting to Docker) and creates a new
container instance. The container is started automatically after creation.
Returns nil if runtime creation, container creation, or container start fails.
*/
func NewBuilder() *Builder {
	errnie.Debug("environment.NewBuilder")

	runtime, err := NewRuntime(Config{
		RuntimeType: "docker", // default to Docker for now
	})
	if err != nil {
		errnie.Error(fmt.Errorf("failed to create runtime: %w", err))
		return nil
	}

	builder := &Builder{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) error {
			errnie.Debug("environment.Builder.buffer.fn")
			return nil
		}),
		Runtime: runtime,
	}

	// Create and start the container
	ctx := context.Background()
	if err := runtime.CreateContainer(ctx); err != nil {
		errnie.Error(fmt.Errorf("failed to create container: %w", err))
		return nil
	}

	if err := runtime.StartContainer(ctx); err != nil {
		errnie.Error(fmt.Errorf("failed to start container: %w", err))
		return nil
	}

	return builder
}

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after builder operations
have been completed.
*/
func (builder *Builder) Read(p []byte) (n int, err error) {
	errnie.Debug("environment.Builder.Read")
	return builder.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing by
the builder.
*/
func (builder *Builder) Write(p []byte) (n int, err error) {
	errnie.Debug("environment.Builder.Write")
	return builder.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (builder *Builder) Close() error {
	errnie.Debug("environment.Builder.Close")
	return builder.buffer.Close()
}
