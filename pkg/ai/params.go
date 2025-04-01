package ai

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ParamsBuilder struct {
	*Params
}

type ParamsOption func(*ParamsBuilder)

func NewParamsBuilder(opts ...ParamsOption) *ParamsBuilder {
	var (
		cpnp   = utils.NewCapnp()
		params Params
		err    error
	)

	if params, err = NewRootParams(cpnp.Seg); errnie.Error(err) != nil {
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

func WithModel(model string) ParamsOption {
	return func(builder *ParamsBuilder) {
		errnie.Error(
			builder.SetModel(model),
		)
	}
}

func WithTemperature(temperature float64) ParamsOption {
	return func(builder *ParamsBuilder) {
		builder.SetTemperature(temperature)
	}
}
