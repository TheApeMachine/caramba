package process

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/core"
)

/*
Plan represents a series of steps to be executed as part of a process.
*/
type Plan struct {
	*core.BaseComponent
	Steps []Step `json:"steps" jsonschema:"description=The steps to execute,required"`
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
	return &Plan{
		BaseComponent: core.NewBaseComponent("plan", core.TypeProcess),
		Steps:         []Step{},
	}
}

/*
AddStep adds a new step to the plan.
*/
func (p *Plan) AddStep(step string) *Plan {
	p.Steps = append(p.Steps, Step{Step: step})
	return p
}

/*
Read serializes the plan to JSON and writes it to the provided buffer.
*/
func (p *Plan) Read(buf []byte) (n int, err error) {
	data, err := json.Marshal(p)
	if err != nil {
		return 0, err
	}

	n = copy(buf, data)
	if n < len(data) {
		return n, io.ErrShortBuffer
	}

	return n, io.EOF
}

/*
Write updates the plan from JSON data.
*/
func (p *Plan) Write(data []byte) (n int, err error) {
	if err = json.Unmarshal(data, p); err != nil {
		return 0, err
	}

	return len(data), nil
}

/*
Close performs any necessary cleanup.
*/
func (p *Plan) Close() error {
	return nil
}
