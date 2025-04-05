package twoface

import "github.com/theapemachine/caramba/pkg/datura"

/*
Cause is a conditional trigger for an Effect, based on the properties of an
Artifact. This allows for a generic way to trigger behavior in a distributed
system, without having to know the specific details of the Artifact.
*/
type Cause struct {
	Role   datura.ArtifactRole
	Scope  datura.ArtifactScope
	Topic  string
	From   string
	To     string
	Effect *Effect
}

type CauseOption func(*Cause)

func NewCause(options ...CauseOption) *Cause {
	cause := &Cause{}

	for _, option := range options {
		option(cause)
	}

	return cause
}

func (cause *Cause) Check(artifact *datura.ArtifactBuilder) bool {
	if cause.Role != datura.ArtifactRoleUnknown && cause.Role != datura.ArtifactRole(artifact.Role()) {
		return false
	}

	if cause.Scope != datura.ArtifactScopeUnknown && cause.Scope != datura.ArtifactScope(artifact.Scope()) {
		return false
	}

	if cause.Topic != "" && cause.Topic != datura.GetMetaValue[string](artifact, "topic") {
		return false
	}

	if cause.From != "" && cause.From != datura.GetMetaValue[string](artifact, "from") {
		return false
	}

	if cause.To != "" && cause.To != datura.GetMetaValue[string](artifact, "to") {
		return false
	}

	return true
}

func WithRole(role datura.ArtifactRole) CauseOption {
	return func(cause *Cause) {
		cause.Role = role
	}
}

func WithScope(scope datura.ArtifactScope) CauseOption {
	return func(cause *Cause) {
		cause.Scope = scope
	}
}

func WithTopic(topic string) CauseOption {
	return func(cause *Cause) {
		cause.Topic = topic
	}
}

func WithFrom(from string) CauseOption {
	return func(cause *Cause) {
		cause.From = from
	}
}

func WithTo(to string) CauseOption {
	return func(cause *Cause) {
		cause.To = to
	}
}

func WithEffect(effect *Effect) CauseOption {
	return func(cause *Cause) {
		cause.Effect = effect
	}
}
