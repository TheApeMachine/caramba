package datura

import (
	"errors"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func (artifact *Artifact) ToState(state errnie.State) *Artifact {
	artifact.SetState(uint64(state))
	return artifact
}

func (artifact *Artifact) Is(state errnie.State) (ok bool) {
	ok = artifact.State() == uint64(state)
	return
}

func (artifact *Artifact) ID() string {
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

func (artifact *Artifact) HasError() bool {
	return artifact.ActsAs(ArtifactRoleAcknowledger) && artifact.ScopedAs(ArtifactScopeError)
}

func (artifact *Artifact) Error() string {
	return string(errnie.Try(artifact.Payload()))
}
