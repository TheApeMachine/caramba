package memory

import "github.com/theapemachine/caramba/pkg/stream"

type Store interface {
	stream.Generator
	Name() string
}
