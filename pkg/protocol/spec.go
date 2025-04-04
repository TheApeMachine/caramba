package protocol

import (
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Spec defines a complete protocol specification that orchestrates the communication
between two agents. It maintains the protocol's state, steps, and conditions while
managing the flow of artifacts through the protocol.

A protocol specification is uniquely identified and tracks both the initiating and
participating agents, along with the current execution state.
*/
type Spec struct {
	ID          string       // Unique identifier for the protocol instance
	initiator   string       // ID of the agent that initiated the protocol
	participant string       // ID of the agent participating in the protocol
	steps       []*Step      // Ordered sequence of protocol steps
	conditions  []*Condition // Active conditions that need to be satisfied
	status      core.Status  // Current status of the protocol
	ptr         int          // Current step pointer in the protocol execution
}

// SpecOption defines a function type for configuring a Spec.
// It follows the functional options pattern for flexible protocol specification.
type SpecOption func(*Spec)

/*
NewSpec creates a new protocol specification with the provided options.
It initializes a new Spec with a unique ID and default values, then applies
any provided configuration options.

Parameters:
  - opts: Variadic slice of SpecOption functions to configure the specification

Returns a pointer to the configured Spec.
*/
func NewSpec(opts ...SpecOption) *Spec {
	errnie.Debug("core.NewSpec")

	spec := &Spec{
		ID:     uuid.New().String(),
		steps:  make([]*Step, 0),
		status: core.StatusUnknown,
		ptr:    -1,
	}

	for _, opt := range opts {
		opt(spec)
	}

	return spec
}

/*
Next advances the protocol to its next step and processes the provided artifact.
It increments the internal step pointer and executes the current step's logic
on the artifact, updating the protocol's state accordingly.

Parameters:
  - artifact: The artifact to process in this step

Returns:
  - The processed artifact
  - The updated status of the protocol
*/
func (spec *Spec) Next(artifact *datura.Artifact) (*datura.Artifact, core.Status) {
	errnie.Debug("core.Spec.Next")

	spec.ptr++

	if spec.ptr >= len(spec.steps) {
		return artifact, core.StatusDone
	}

	step := spec.steps[spec.ptr]
	artifact, spec.status, spec.conditions = step.Do(
		spec.ID,
		artifact,
		spec.status,
		spec.conditions,
	)

	return artifact, spec.status
}

/*
WithInitiator is a SpecOption that sets the initiating agent's ID for the protocol.
The initiator is the agent that starts the protocol execution.

Parameters:
  - initiator: The ID of the initiating agent
*/
func WithInitiator(initiator string) SpecOption {
	return func(spec *Spec) {
		spec.initiator = initiator
	}
}

/*
WithParticipant is a SpecOption that sets the participating agent's ID for the protocol.
The participant is the agent that responds to the protocol initiation.

Parameters:
  - participant: The ID of the participating agent
*/
func WithParticipant(participant string) SpecOption {
	return func(spec *Spec) {
		spec.participant = participant
	}
}

/*
WithSteps is a SpecOption that adds steps to the protocol specification.
Steps are executed in the order they are provided and define the protocol's flow.

Parameters:
  - steps: Variadic slice of Step pointers to add to the specification
*/
func WithSteps(steps ...*Step) SpecOption {
	return func(spec *Spec) {
		spec.steps = append(spec.steps, steps...)
	}
}

/*
WithStatus is a SpecOption that sets the initial status of the protocol.
This defines the starting state of the protocol before execution begins.

Parameters:
  - status: The initial Status to set for the protocol
*/
func WithStatus(status core.Status) SpecOption {
	return func(spec *Spec) {
		spec.status = status
	}
}
