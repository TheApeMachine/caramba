package protocol

import (
	"slices"

	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Step represents a single action or state transition within a protocol.
It defines the expected conditions, directions of communication, and the
resulting state changes that should occur during this part of the protocol.

Each step maintains its own set of conditions that must be met, the roles and
scopes of artifacts it handles, and the status it should transition to.
*/
type Step struct {
	Conditions []*Condition         // Conditions that must be satisfied for this step
	Directions []string             // Communication direction indicators (from/to)
	Role       datura.ArtifactRole  // Expected role of artifacts in this step
	Scope      datura.ArtifactScope // Expected scope of artifacts in this step
	Status     core.Status          // Status to transition to after this step
}

// StepOption defines a function type for configuring a Step.
// It follows the functional options pattern for flexible step configuration.
type StepOption func(*Step)

/*
NewStep creates a new protocol step with the provided options.
It initializes a new Step with empty conditions and directions slices,
then applies any provided configuration options.

Parameters:
  - opts: Variadic slice of StepOption functions to configure the step

Returns a pointer to the configured Step.
*/
func NewStep(opts ...StepOption) *Step {
	step := &Step{
		Conditions: make([]*Condition, 0),
		Directions: make([]string, 0),
	}

	for _, opt := range opts {
		opt(step)
	}

	return step
}

/*
Do executes the step's logic on an artifact, applying any necessary transformations
and checking conditions. It manages the protocol's state transition and condition tracking.

Parameters:
  - id: The protocol instance identifier
  - artifact: The artifact to process in this step
  - status: The current status of the protocol
  - conditions: The current set of active conditions

Returns:
  - The processed artifact
  - The new status after processing
  - The updated set of conditions
*/
func (step *Step) Do(
	id string,
	artifact *datura.Artifact,
	status core.Status,
	initiator string,
	participant string,
	initiatorConditions []*Condition,
	participantConditions []*Condition,
) (*datura.Artifact, core.Status, []*Condition, []*Condition) {
	errnie.Debug("core.Step.Do")
	var conditions []*Condition

	if datura.GetMetaValue[string](artifact, "from") != initiator {
		conditions = participantConditions
	}

	if datura.GetMetaValue[string](artifact, "from") != participant {
		conditions = initiatorConditions
	}

	// Resolve any existing conditions.
	for i, condition := range conditions {
		if condition.Check(artifact, status) {
			conditions = slices.Delete(conditions, i, i+1)
			errnie.Info("core.Step.Do.Condition.Resolved", "role", artifact.Role(), "scope", artifact.Scope())
		}
	}

	// Create a new artifact instead of modifying the input one
	response := datura.New(
		datura.WithRole(step.Role),
		datura.WithScope(step.Scope),
	)
	response.SetMetaValue("from", step.Directions[0])
	response.SetMetaValue("to", step.Directions[1])
	response.SetMetaValue("protocol", id)

	return response, step.Status, initiatorConditions, participantConditions
}

/*
WithDirection is a StepOption that sets the communication directions for the step.
Directions typically specify the 'from' and 'to' agents for the communication.

Parameters:
  - directions: Variadic slice of strings defining the communication path
*/
func WithDirection(directions ...string) StepOption {
	return func(step *Step) {
		step.Directions = directions
	}
}

/*
WithStepRole is a StepOption that sets the expected artifact role for the step.
This defines what role an artifact should have when processed by this step.

Parameters:
  - role: The ArtifactRole to set for the step
*/
func WithStepRole(role datura.ArtifactRole) StepOption {
	return func(step *Step) {
		step.Role = role
	}
}

/*
WithStepScope is a StepOption that sets the expected artifact scope for the step.
This defines what scope an artifact should have when processed by this step.

Parameters:
  - scope: The ArtifactScope to set for the step
*/
func WithStepScope(scope datura.ArtifactScope) StepOption {
	return func(step *Step) {
		step.Scope = scope
	}
}

/*
WithStepStatus is a StepOption that sets the target status for the step.
This defines what status the protocol should transition to after this step.

Parameters:
  - status: The Status to set for the step
*/
func WithStepStatus(status core.Status) StepOption {
	return func(step *Step) {
		step.Status = status
	}
}

/*
WithConditions is a StepOption that sets the conditions required for the step.
These conditions must be satisfied for the step to be considered complete.

Parameters:
  - conditions: Variadic slice of Condition pointers to add to the step
*/
func WithConditions(conditions ...*Condition) StepOption {
	return func(step *Step) {
		step.Conditions = conditions
	}
}
