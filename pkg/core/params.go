package core

import (
	capnp "capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/datura"
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

func (builder *ParamsBuilder) Artifact() *datura.Artifact {
	return datura.New(
		datura.WithPayload(builder.Payload()),
		datura.WithRole(datura.ArtifactRoleAnswer),
		datura.WithScope(datura.ArtifactScopeParams),
	)
}

func (builder *ParamsBuilder) Payload() []byte {
	payload, err := builder.Params.Message().Marshal()

	if errnie.Error(err) != nil {
		return nil
	}

	return payload
}

type ResponseFormat struct {
	Name        func() (string, error)
	Description func() (string, error)
	Schema      func() (string, error)
	Strict      func() (bool, error)
}

func (builder *ParamsBuilder) ResponseFormat() *ResponseFormat {
	return &ResponseFormat{
		Name:        func() (string, error) { return "tmp", nil },
		Description: func() (string, error) { return "tmp", nil },
		Schema:      func() (string, error) { return "tmp", nil },
		Strict:      func() (bool, error) { return true, nil },
	}
}

func (builder *ParamsBuilder) WithArtifact(artifact *datura.Artifact) *ParamsBuilder {
	payload, err := artifact.DecryptPayload()

	if errnie.Error(err) != nil {
		return builder
	}

	msgData, err := capnp.Unmarshal(payload)

	if errnie.Error(err) != nil {
		return builder
	}

	msg, err := ReadRootParams(msgData)

	if errnie.Error(err) != nil {
		return builder
	}

	builder.Params = &msg
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
