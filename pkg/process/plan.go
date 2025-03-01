package process

import (
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/agent/util"
)

type Plan struct {
	Goal  string     `json:"goal" jsonschema:"description=The goal of the plan,required"`
	Steps []PlanStep `json:"steps" jsonschema:"description=The steps to complete the plan,required"`
}

type PlanStep struct {
	StepID             string               `json:"step_id" jsonschema:"description=The unique identifier for the step,required"`
	Order              int                  `json:"order" jsonschema:"description=The order of the step in the plan,required"`
	Description        string               `json:"description" jsonschema:"description=The description of the step,required"`
	AcceptanceCriteria []AcceptanceCriteria `json:"acceptance_criteria" jsonschema:"description=The acceptance criteria for the step,required"`
}

type AcceptanceCriteria struct {
	Criteria string `json:"criteria" jsonschema:"description=The criteria for the step,required"`
}

func (plan *Plan) Name() string {
	return "Plan"
}

func (plan *Plan) Description() string {
	return "A plan to complete a task"
}

func (plan *Plan) Schema() any {
	return util.GenerateSchema[Plan]()
}

/*
String converts the structured output to a simple Markdown structured string.
*/
func (plan *Plan) String() string {
	builder := strings.Builder{}

	for _, step := range plan.Steps {
		builder.WriteString(fmt.Sprintf("## Step %d\n", step.Order))
		builder.WriteString(fmt.Sprintf("### Description\n%s\n", step.Description))
		builder.WriteString("### Acceptance Criteria\n")

		for _, criteria := range step.AcceptanceCriteria {
			builder.WriteString(fmt.Sprintf("- %s\n", criteria.Criteria))
		}
	}

	return builder.String()
}
