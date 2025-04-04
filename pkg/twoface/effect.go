package twoface

import "github.com/theapemachine/caramba/pkg/datura"

/*
Effect is a function that is executed when a Cause is triggered. It is used to
modify the Artifact in some way, or to trigger other effects. It is also a
Job type, so it can be scheduled onto a worker pool.
*/
type Effect struct {
	fn func(*datura.Artifact, chan *datura.Artifact) *datura.Artifact
}

type EffectOption func(*Effect)

func NewEffect(options ...EffectOption) *Effect {
	effect := &Effect{}

	for _, option := range options {
		option(effect)
	}

	return effect
}

func (effect *Effect) Do(artifact *datura.Artifact, out chan *datura.Artifact) *datura.Artifact {
	return effect.fn(artifact, out)
}

func WithFunction(fn func(*datura.Artifact, chan *datura.Artifact) *datura.Artifact) EffectOption {
	return func(effect *Effect) {
		effect.fn = fn
	}
}
