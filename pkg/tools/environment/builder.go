package environment

import (
	"context"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Builder manages the creation and initialization of container environments.
It handles container runtime setup and provides a buffered interface for
environment operations.
*/
type Builder struct {
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
		errnie.New(errnie.WithError(err))
		return nil
	}

	// Create and start the container
	ctx := context.Background()
	if err := runtime.CreateContainer(ctx); err != nil {
		errnie.New(errnie.WithError(err))
		return nil
	}

	if err := runtime.StartContainer(ctx); err != nil {
		errnie.New(errnie.WithError(err))
		return nil
	}

	return &Builder{
		Runtime: runtime,
	}
}
