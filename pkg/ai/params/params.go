package params

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type ParamsBuilder struct {
	Params *Params
}

type ParamsOption func(*ParamsBuilder)

func New(opts ...ParamsOption) *ParamsBuilder {
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

	if err := params.SetModel(
		tweaker.GetModel(tweaker.GetProvider()),
	); errnie.Error(err) != nil {
		return nil
	}

	builder := &ParamsBuilder{
		Params: &params,
	}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}
