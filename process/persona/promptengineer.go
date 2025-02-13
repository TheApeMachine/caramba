package persona

import (
	"github.com/theapemachine/amsh/utils"
)

type PromptEngineer struct {
	Goals       []string `json:"goals" jsonschema:"description=The goals you are able to detect in the user prompt,required"`
	FinalAnswer string   `json:"final_answer" jsonschema:"description=The final optimized prompt that is sent to the next agent"`
}

func (promptEngineer *PromptEngineer) GenerateSchema() string {
	return utils.GenerateSchema[PromptEngineer]()
}

func (promptEngineer *PromptEngineer) Name() string {
	return "Prompt Engineer"
}

func (promptEngineer *PromptEngineer) Description() string {
	return "An expert in refining prompts to be optimized for LLM processing."
}
