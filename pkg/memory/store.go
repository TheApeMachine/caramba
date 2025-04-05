package memory

import "github.com/theapemachine/caramba/pkg/datura"

type Store interface {
	Generate(
		buffer chan *datura.ArtifactBuilder,
		fn ...func(artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder,
	) chan *datura.ArtifactBuilder
	Name() string
}
