/*
Package protocol provides primitives and utilities for defining and executing
communication protocols between agents in the Caramba system.
*/
package protocol

import (
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Condition represents a set of criteria that must be met for a protocol step to be
considered satisfied. It combines role, scope, and status checks for artifacts.

A condition is used to validate the state of an artifact at a particular point
in the protocol execution flow.
*/
type Condition struct {
	Role   datura.ArtifactRole
	Scope  datura.ArtifactScope
	Status core.Status
}

// ConditionOption defines a function type for configuring a Condition.
// It follows the functional options pattern for flexible condition configuration.
type ConditionOption func(*Condition)

/*
NewCondition creates a new Condition instance with the provided options.
It uses the functional options pattern to allow flexible configuration of the condition.

Parameters:
  - opts: Variadic slice of ConditionOption functions to configure the condition

Returns a pointer to the configured Condition.
*/
func NewCondition(opts ...ConditionOption) *Condition {
	condition := &Condition{}

	for _, opt := range opts {
		opt(condition)
	}

	return condition
}

/*
Check validates whether an artifact meets all the criteria specified in the condition.
It compares the artifact's role, scope, and the current status against the condition's requirements.

Parameters:
  - artifact: The artifact to check against the condition
  - status: The current status to compare against the condition's status

Returns true if all criteria are met, false otherwise.
*/
func (condition *Condition) Check(artifact *datura.Artifact, status core.Status) bool {
	if condition.Role != datura.ArtifactRole(artifact.Role()) {
		errnie.Warn("protocol.Condition.Check.Role", "role", artifact.Role(), "expected", condition.Role)
		return false
	}

	errnie.Info("protocol.Condition.Check.Role.resolved")

	if condition.Scope != datura.ArtifactScope(artifact.Scope()) {
		errnie.Warn("protocol.Condition.Check.Scope", "scope", artifact.Scope(), "expected", condition.Scope)
		return false
	}

	errnie.Info("protocol.Condition.Check.Scope.resolved")

	if condition.Status != status {
		errnie.Warn("protocol.Condition.Check.Status", "status", status, "expected", condition.Status)
		return false
	}

	errnie.Info("protocol.Condition.Check.Status.resolved")

	return true
}

/*
WithConditionRole is a ConditionOption that sets the role requirement for a condition.
The role specifies the expected role of an artifact for the condition to be satisfied.

Parameters:
  - role: The ArtifactRole to set for the condition
*/
func WithConditionRole(role datura.ArtifactRole) ConditionOption {
	return func(condition *Condition) {
		condition.Role = role
	}
}

/*
WithConditionScope is a ConditionOption that sets the scope requirement for a condition.
The scope defines the expected scope of an artifact for the condition to be satisfied.

Parameters:
  - scope: The ArtifactScope to set for the condition
*/
func WithConditionScope(scope datura.ArtifactScope) ConditionOption {
	return func(condition *Condition) {
		condition.Scope = scope
	}
}

/*
WithConditionStatus is a ConditionOption that sets the status requirement for a condition.
The status defines the expected status for the condition to be satisfied.

Parameters:
  - status: The Status to set for the condition
*/
func WithConditionStatus(status core.Status) ConditionOption {
	return func(condition *Condition) {
		condition.Status = status
	}
}
