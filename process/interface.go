package process

import "github.com/theapemachine/caramba/datura"

type Interface interface {
	Name() string
	Description() string
	Schema() interface{}
	Process(input *datura.Artifact) chan *datura.Artifact
}
