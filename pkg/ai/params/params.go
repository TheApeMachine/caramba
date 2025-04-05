package params

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ParamsBuilder struct {
	Params *Params
}

func New() *ParamsBuilder {
	var (
		arena  = capnp.SingleSegment(nil)
		seg    *capnp.Segment
		params Params
		err    error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if params, err = NewRootParams(seg); errnie.Error(err) != nil {
		return nil
	}

	return &ParamsBuilder{
		Params: &params,
	}
}
