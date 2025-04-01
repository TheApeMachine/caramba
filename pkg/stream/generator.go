package stream

import "github.com/theapemachine/caramba/pkg/datura"

type Generator interface {
	Generate(chan *datura.Artifact) chan *datura.Artifact
}
