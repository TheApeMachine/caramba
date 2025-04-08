package datura

import (
	"errors"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (artifact *Artifact) ToState(state errnie.State) *Artifact {
	errnie.Trace("artifact.ToState", "id", artifact.ID(), "state", state)
	artifact.SetState(uint64(state))
	errnie.Debug("artifact.ToState", "id", artifact.ID(), "state", state)
	return artifact
}

func (artifact *Artifact) Is(state errnie.State) (ok bool) {
	ok = artifact.State() == uint64(state)
	errnie.Debug("artifact.Is", "id", artifact.ID(), "state", state, "ok", ok)
	return
}

func (artifact *Artifact) ID() string {
	errnie.Trace("artifact.ID", "id", errnie.Try(artifact.Uuid()))
	uid := errnie.Try(artifact.Uuid())

	if uid == "" {
		if errnie.Error(artifact.SetUuid(uuid.New().String())) != nil {
			errnie.Fatal(errnie.New(
				errnie.WithError(errors.New("failed to set uuid")),
				errnie.WithMessage("failed to set uuid"),
			))
		}
	}

	return uid
}
