package optimize

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func NewCapnpOptimize() (*Optimize, error) {
	var (
		arena    = capnp.SingleSegment(nil)
		seg      *capnp.Segment
		optimize Optimize
		err      error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil, errnie.Error(err)
	}

	if optimize, err = NewRootOptimize(seg); errnie.Error(err) != nil {
		return nil, errnie.Error(err)
	}

	return &optimize, nil
}
