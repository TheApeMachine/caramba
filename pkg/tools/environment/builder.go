package environment

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Builder struct {
	buffer  *stream.Buffer
	Runtime Runtime
}

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

func (builder *Builder) Read(p []byte) (n int, err error) {
	errnie.Debug("environment.Builder.Read")

	return builder.buffer.Read(p)
}

func (builder *Builder) Write(p []byte) (n int, err error) {
	errnie.Debug("environment.Builder.Write")

	return builder.buffer.Write(p)
}

func (builder *Builder) Close() error {
	errnie.Debug("environment.Builder.Close")

	return builder.buffer.Close()
}
