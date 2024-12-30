package prompt

import "github.com/theapemachine/amsh/utils"

type Process struct {
	Keywords    []string `json:"keywords" jsonschema:"description=The keywords extracted from the prompt,required"`
	Objectives  []string `json:"objectives" jsonschema:"description=The objectives extracted from the prompt,required"`
	Ambiguities []string `json:"ambiguities" jsonschema:"description=The ambiguities extracted from the prompt,required"`
	FinalPrompt string   `json:"final_prompt" jsonschema:"description=The final re-engineered, optimized prompt,required"`
}

func (p *Process) Name() string {
	return "Prompt"
}

func (p *Process) Description() string {
	return "A structured prompt (re)engineering process to improve the prompt's clarity, specificity, and effectiveness."
}

func (p *Process) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Process]()
}
