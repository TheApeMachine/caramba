package mechanic

import "github.com/theapemachine/caramba/utils"

type Process struct {
	Metrics []Metric `json:"metrics" jsonschema:"title=Metrics,description=Metrics to monitor,required"`
}

type Metric struct {
	Name           string          `json:"name" jsonschema:"title=Name,description=The name of the agent you are evaluating,required"`
	Role           string          `json:"role" jsonschema:"title=Role,description=The role of the agent you are evaluating,required"`
	Observations   []Observation   `json:"observations" jsonschema:"title=Observations,description=Observations of the agent you are evaluating,required"`
	Considerations []Consideration `json:"considerations" jsonschema:"title=Considerations,description=Additional considerations before making any changes,required"`
	Changes        []Change        `json:"changes" jsonschema:"title=Changes,description=Changes to make to the agent you are evaluating"`
	FineTuning     []FineTuning    `json:"fineTuning" jsonschema:"title=FineTuning,description=High quality instruction -> response pairs to use for fine-tuning"`
}

type Observation struct {
	Metric string `json:"metric" jsonschema:"title=Metric,description=The metric you are observing,required"`
	Value  string `json:"value" jsonschema:"title=Value,description=The value of the metric,required"`
}

type Consideration struct {
	Consideration string `json:"consideration" jsonschema:"title=Consideration,description=The consideration you are making,required"`
	Impact        string `json:"impact" jsonschema:"title=Impact,description=The impact of the consideration,required"`
}

type Change struct {
	SystemPrompt string `json:"systemPrompt" jsonschema:"title=SystemPrompt,description=The new system prompt you are setting"`
	Temperature  string `json:"temperature" jsonschema:"title=Temperature,description=The new temperature you are setting"`
	TopP         string `json:"topP" jsonschema:"title=TopP,description=The new topP you are setting"`
	TopK         string `json:"topK" jsonschema:"title=TopK,description=The new topK you are setting"`
}

type FineTuning struct {
	Instruction string `json:"instruction" jsonschema:"title=Instruction,description=The instruction that was provided to the agent,required"`
	Response    string `json:"response" jsonschema:"title=Response,description=The response that the agent provided,required"`
}

func (process *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
