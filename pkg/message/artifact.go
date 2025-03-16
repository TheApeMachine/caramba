package message

import (
	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func root() (*Artifact, error) {
	arena := capnp.SingleSegment(nil)

	_, seg, err := capnp.NewMessage(arena)
	if errnie.Error(err) != nil {
		return nil, err
	}

	artfct, err := NewRootArtifact(seg)
	if errnie.Error(err) != nil {
		return nil, err
	}

	return &artfct, nil
}

/*
New creates a new artifact with the given origin, role, scope, and data.
*/
func New(role Role, name, content string) *Artifact {
	var (
		err      error
		artifact *Artifact
	)

	if artifact, err = root(); errnie.Error(err) != nil {
		return nil
	}

	artifact.SetId(uuid.New().String())

	if err := artifact.SetName(name); err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetRole(role.String()); err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetContent(content); err != nil {
		errnie.Error(err)
		return nil
	}

	return artifact
}
