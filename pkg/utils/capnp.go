package utils

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Capnp struct {
	Arena *capnp.SingleSegmentArena
	Seg   *capnp.Segment
	Err   error
}

func NewCapnp() *Capnp {
	arena := capnp.SingleSegment(nil)

	cpnp := &Capnp{
		Arena: arena,
	}

	if _, cpnp.Seg, cpnp.Err = capnp.NewMessage(
		cpnp.Arena,
	); errnie.Error(cpnp.Err) != nil {
		return nil
	}

	return cpnp
}
