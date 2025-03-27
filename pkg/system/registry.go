package system

import (
	"io"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/workflow"
)

var (
	once     sync.Once
	registry *Registry
)

type Registry struct {
	buffer     *stream.Buffer
	components *map[uint32]map[uint32][]io.ReadWriteCloser
}

func NewRegistry() *Registry {
	once.Do(func() {
		components := map[uint32]map[uint32][]io.ReadWriteCloser{}

		registry = &Registry{
			buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
				errnie.Debug("system.NewRegistry.buffer.fn")

				switch artifact.Role() {
				case uint32(datura.ArtifactRoleRegistration):
					
				default:
					out := make([]string, 0)

					for _, component := range components[artifact.Role()][artifact.Scope()] {
						if err = workflow.NewFlipFlop(artifact, component); err != nil {
							out = append(out, errnie.Error(err).Error())
							continue
						}

						out = append(out, datura.GetMetaValue[string](artifact, "output"))
					}

					artifact.SetMetaValue("output", strings.Join(out, "\n\n"))
				}

				return nil
			}),
			components: &components,
		}
	})

	return registry
}

func (registry *Registry) Read(p []byte) (n int, err error) {
	return registry.buffer.Read(p)
}

func (registry *Registry) Write(p []byte) (n int, err error) {
	return registry.buffer.Write(p)
}

func (registry *Registry) Close() error {
	return registry.buffer.Close()
}
