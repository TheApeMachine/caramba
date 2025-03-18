package process

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
Plan represents a series of steps to be executed as part of a process.
*/
type Plan struct {
	Steps []Step `json:"steps" jsonschema:"description=The steps to take,required"`
}

/*
Step represents a single action to be taken in a plan.
*/
type Step struct {
	Step string `json:"step" jsonschema:"description=The step to execute,required"`
}

/*
NewPlan creates a new plan with initialized components.
*/
func NewPlan() *Plan {
	errnie.Debug("NewPlan")
	return &Plan{}
}

func (proc *Plan) Name() string {
	return "plan"
}

func (proc *Plan) Description() string {
	return "A plan is a series of steps to be executed as part of a process."
}

func (proc *Plan) Schema() any {
	return utils.GenerateSchema[Plan]()
}

func (proc *Plan) Strict() bool {
	return true
}
