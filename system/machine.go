package system

import "github.com/theapemachine/caramba/stream"

type Machine struct {
	environments []stream.Generator
}

func NewMachine() *Machine {
	return &Machine{}
}
