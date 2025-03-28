package system

import (
	"errors"
	"io"
	"sync"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type ComponentType uint

const (
	ComponentTypeUnknown ComponentType = iota
	ComponentTypeAgent
)

var (
	once     sync.Once
	registry *Registry
)

type Registry struct {
	buffer     *stream.Buffer
	components map[ComponentType][]io.ReadWriteCloser
}

type RegistryOption func(*Registry)

func NewRegistry(opts ...RegistryOption) *Registry {
	errnie.Debug("system.NewRegistry")

	once.Do(func() {
		registry = &Registry{
			buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
				errnie.Debug("system.NewRegistry.buffer.fn")

				role := artifact.Role()

				switch role {
				case uint32(datura.ArtifactRoleSignal):
					command := datura.GetMetaValue[string](artifact, "command")

					switch command {
					case "inspect":
						inspectArtifact := datura.New(
							datura.WithRole(datura.ArtifactRoleInspect),
							datura.WithScope(datura.ArtifactScopeName),
						)

						for _, component := range registry.components[ComponentTypeAgent] {
							workflow.NewFlipFlop(inspectArtifact, component)
						}

						artifact.SetMetaValue("to", datura.GetMetaValue[string](inspectArtifact, "name"))
					case "send":
						message := datura.GetMetaValue[string](artifact, "message_arg")

						for _, component := range registry.components[ComponentTypeAgent] {
							datura.WithPayload([]byte(message))(artifact)
							workflow.NewFlipFlop(artifact, component)
						}
					}
				}

				return nil
			}),
			components: make(map[ComponentType][]io.ReadWriteCloser),
		}

		for _, opt := range opts {
			opt(registry)
		}
	})

	return registry
}

func (registry *Registry) Read(p []byte) (n int, err error) {
	errnie.Debug("system.Registry.Read")

	if registry.buffer == nil {
		return 0, errnie.Error(errors.New("buffer not set"))
	}

	return registry.buffer.Read(p)
}

func (registry *Registry) Write(p []byte) (n int, err error) {
	errnie.Debug("system.Registry.Write")

	if registry.buffer == nil {
		return 0, errnie.Error(errors.New("buffer not set"))
	}

	return registry.buffer.Write(p)
}

func (registry *Registry) Close() error {
	errnie.Debug("system.Registry.Close")

	if registry.buffer == nil {
		return errnie.Error(errors.New("buffer not set"))
	}

	return registry.buffer.Close()
}

func WithComponents(components map[ComponentType][]io.ReadWriteCloser) RegistryOption {
	return func(registry *Registry) {
		registry.components = components
	}
}

func WithComponent(componentType ComponentType, component io.ReadWriteCloser) RegistryOption {
	return func(registry *Registry) {
		if _, ok := registry.components[componentType]; !ok {
			registry.components[componentType] = make([]io.ReadWriteCloser, 0)
		}

		registry.components[componentType] = append(registry.components[componentType], component)
	}
}
