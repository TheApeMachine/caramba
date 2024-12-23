package prompt

import "github.com/theapemachine/amsh/utils"

type Process struct {
	Assessment []Assessment `json:"assessment" jsonschema:"title=Assessment,description=The assessment of the prompt and context,required"`
	Improved   string       `json:"improved" jsonschema:"title=Improved,description=The improved and cleaned context,required"`
}

type Assessment struct {
	Irrelevant []string `json:"irrelevant" jsonschema:"title=Irrelevant,description=Irrelevant parts of the current context,required"`
	Incorrect  []string `json:"incorrect" jsonschema:"title=Incorrect,description=Incorrect parts of the current context,required"`
	Confusing  []string `json:"confusing" jsonschema:"title=Confusing,description=Confusing parts of the current context,required"`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
