package environment

import (
	"fmt"

	"github.com/containerd/containerd/v2/client"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Builder struct {
	buffer    *stream.Buffer
	client    *client.Client
	Container *Container
}

func NewBuilder() *Builder {
	errnie.Debug("environment.NewBuilder")

	conn, err := client.New("/var/run/containerd/containerd.sock", client.WithDefaultNamespace("caramba"))

	if errnie.Error(err) != nil {
		return nil
	}

	var container *Container

	if container = NewContainer(conn); container == nil {
		errnie.Error(fmt.Errorf("container is nil"))
		return nil
	}

	builder := &Builder{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) error {
			errnie.Debug("environment.Builder.buffer.fn")

			if errnie.Error(container.Load()) != nil {
				return err
			}

			return nil
		}),
		client:    conn,
		Container: container,
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
