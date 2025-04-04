package ai

import (
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/protocol"
)

type Task struct {
	ID          string
	Initiator   string
	Participant string
}

func NewTask(initiator string, participant string) *protocol.Spec {
	return protocol.NewSpec(
		protocol.WithSteps(
			protocol.NewStep(
				protocol.WithDirection(initiator, participant),
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
				protocol.WithDirection(participant, initiator),
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
				protocol.WithDirection(initiator, participant),
				protocol.WithStepRole(datura.ArtifactRoleAcknowledge),
				protocol.WithStepScope(datura.ArtifactScopeAquire),
				protocol.WithStepStatus(core.StatusBusy),
			),
			protocol.NewStep(
				protocol.WithDirection(participant, initiator),
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

var protocols = map[string]func(string, string) *protocol.Spec{
	"task": NewTask,
}
