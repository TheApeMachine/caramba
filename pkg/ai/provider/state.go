package provider

import "github.com/theapemachine/caramba/pkg/errnie"

func (prvdr Provider) Is(state errnie.State) bool {
	errnie.Trace("provider.Is", "id", prvdr.ID(), "state", state)

	return prvdr.State() == uint64(state)
}

func (prvdr Provider) ToState(state errnie.State) Provider {
	errnie.Trace("provider.ToState", "id", prvdr.ID(), "state", state)

	prvdr.SetState(uint64(state))
	return prvdr
}

func (prvdr Provider) ID() string {
	errnie.Trace("provider.ID", "id", errnie.Try(prvdr.Uuid()))

	return errnie.Try(prvdr.Uuid())
}
