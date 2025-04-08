package datura

import "github.com/theapemachine/caramba/pkg/errnie"

func (artifact Artifact) ToState(state errnie.State) Artifact {
	artifact.SetState(uint64(state))
	return artifact
}

func (artifact Artifact) Is(state errnie.State) bool {
	return artifact.State() == uint64(state)
}

func (artifact Artifact) ID() string {
	return errnie.Try(artifact.Uuid())
}
