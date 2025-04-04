package ai

import (
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/protocol"
)

type Protocols struct {
	Initiator   string
	Participant string
}

type ProtocolOption func(*Protocols)

func NewProtocols(opts ...ProtocolOption) *Protocols {
	protocols := &Protocols{}

	for _, opt := range opts {
		opt(protocols)
	}

	return protocols
}

func (protocols *Protocols) Task(id string) *protocol.Spec {
	return protocol.NewSpec(
		protocol.WithInitiator(protocols.Initiator),
		protocol.WithParticipant(protocols.Participant),
		protocol.WithSteps(
			protocol.NewStep(
				protocol.WithDirection(protocols.Initiator, protocols.Participant),
				protocol.WithStepRole(datura.ArtifactRoleQuestion),
				protocol.WithStepScope(datura.ArtifactScopeAquire),
				protocol.WithStepStatus(core.StatusWaiting),
				protocol.WithConditions(
					protocol.NewCondition(
						protocol.WithConditionRole(datura.ArtifactRoleAcknowledge),
						protocol.WithConditionScope(datura.ArtifactScopeAquire),
						protocol.WithConditionStatus(core.StatusWaiting),
					),
				),
			),
			protocol.NewStep(
				protocol.WithDirection(protocols.Participant, protocols.Initiator),
				protocol.WithStepRole(datura.ArtifactRoleAcknowledge),
				protocol.WithStepScope(datura.ArtifactScopeAquire),
				protocol.WithStepStatus(core.StatusWaiting),
				protocol.WithConditions(
					protocol.NewCondition(
						protocol.WithConditionRole(datura.ArtifactRoleAcknowledge),
						protocol.WithConditionScope(datura.ArtifactScopeAquire),
						protocol.WithConditionStatus(core.StatusWaiting),
					),
				),
			),
			protocol.NewStep(
				protocol.WithDirection(protocols.Initiator, protocols.Participant),
				protocol.WithStepRole(datura.ArtifactRoleAcknowledge),
				protocol.WithStepScope(datura.ArtifactScopeAquire),
				protocol.WithStepStatus(core.StatusBusy),
			),
			protocol.NewStep(
				protocol.WithDirection(protocols.Participant, protocols.Initiator),
				protocol.WithStepRole(datura.ArtifactRoleQuestion),
				protocol.WithStepScope(datura.ArtifactScopePreflight),
				protocol.WithStepStatus(core.StatusWaiting),
				protocol.WithConditions(
					protocol.NewCondition(
						protocol.WithConditionRole(datura.ArtifactRoleAcknowledge),
						protocol.WithConditionScope(datura.ArtifactScopeParams),
						protocol.WithConditionStatus(core.StatusWaiting),
					),
					protocol.NewCondition(
						protocol.WithConditionRole(datura.ArtifactRoleAcknowledge),
						protocol.WithConditionScope(datura.ArtifactScopeContext),
						protocol.WithConditionStatus(core.StatusWaiting),
					),
				),
			),
		),
	)
}

func WithInitiator(initiator string) ProtocolOption {
	return func(protocols *Protocols) {
		protocols.Initiator = initiator
	}
}

func WithParticipant(participant string) ProtocolOption {
	return func(protocols *Protocols) {
		protocols.Participant = participant
	}
}

var protocols = map[string]func(string) *protocol.Spec{
	"task": NewProtocols(
		WithInitiator("user"),
		WithParticipant("agent"),
	).Task,
}
