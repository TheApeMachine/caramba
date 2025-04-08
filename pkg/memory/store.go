package memory

import "github.com/theapemachine/caramba/pkg/datura"

type Store interface {
	Generate(
		buffer chan *datura.Artifact,
		fn ...func(artifact *datura.Artifact) *datura.Artifact,
	) chan *datura.Artifact
	Name() string
}
