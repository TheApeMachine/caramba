package tool

import (
	"capnproto.org/go/capnp/v3"
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
	function Function,
) *Artifact {
	var (
		err      error
		artifact *Artifact
	)

	if artifact, err = root(); errnie.Error(err) != nil {
		return nil
	}

	artifact.SetType("function")
	artifact.SetFunction(function)

	return artifact
}
