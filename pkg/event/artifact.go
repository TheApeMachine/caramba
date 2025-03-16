package event

import (
	"time"

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
func New(
	origin string, typ Type, role Role, payload []byte,
) *Artifact {
	var (
		err      error
		artifact *Artifact
	)

	if artifact, err = root(); errnie.Error(err) != nil {
		return nil
	}

	artifact.SetId(uuid.New().String())
	artifact.SetTimestamp(time.Now().UnixNano())

	if err := artifact.SetType(typ.String()); err != nil {
		errnie.Error(err)
		return nil
	}

	// Error handling: if setting any required field fails, return Empty()
	if err := artifact.SetOrigin(origin); err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetRole(role.String()); err != nil {
		errnie.Error(err)
		return nil
	}

	if err := artifact.SetPayload(payload); err != nil {
		errnie.Error(err)
		return nil
	}

	return artifact
}
