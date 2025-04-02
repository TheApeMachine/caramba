package stream

import "github.com/theapemachine/caramba/pkg/datura"

type Generator interface {
	Generate(
		chan *datura.Artifact,
		...func(*datura.Artifact) *datura.Artifact,
	) chan *datura.Artifact
}
