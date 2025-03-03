package process

import (
	"fmt"
	"strings"

	"github.com/theapemachine/amsh/utils"
)

type Optimize struct {
	Observations    []Observation    `json:"observations" jsonschema:"description=Observations from the previous process,required"`
	Recommendations []Recommendation `json:"recommendations" jsonschema:"description=Recommendations for the next process,required"`
}

type Observation struct {
	Step   string `json:"step" jsonschema:"description=The step that was executed,required"`
	Result string `json:"result" jsonschema:"description=The result of the step,required"`
}

type Recommendation struct {
	Tip    string `json:"tip" jsonschema:"description=A tip for the next process,required"`
	Reason string `json:"reason" jsonschema:"description=The reason for the recommendation,required"`
}

func (o *Optimize) Name() string {
	return "Optimize"
}

func (o *Optimize) Description() string {
	return "Optimize the plan"
}

func (o *Optimize) Schema() any {
	return utils.GenerateSchema[Optimize]()
}

func (o *Optimize) String() string {
	builder := strings.Builder{}

	for _, observation := range o.Observations {
		builder.WriteString(fmt.Sprintf("Observation: %s\n", observation.Step))
		builder.WriteString(fmt.Sprintf("Result: %s\n", observation.Result))
	}

	for _, recommendation := range o.Recommendations {
		builder.WriteString(fmt.Sprintf("Recommendation: %s\n", recommendation.Tip))
		builder.WriteString(fmt.Sprintf("Reason: %s\n", recommendation.Reason))
	}

	return builder.String()
}
