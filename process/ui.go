package process

import (
	"github.com/theapemachine/caramba/utils"
)

type UI struct {
	OriginalPrompt  string    `json:"original_prompt" jsonschema:"description=The original prompt from the user,required"`
	Reasoning       []Thought `json:"reasoning" jsonschema:"description=The reasoning for the optimized prompt,required"`
	OptimizedPrompt string    `json:"optimized_prompt" jsonschema:"description=The optimized prompt from the user,required"`
	FinalResponse   string    `json:"final_response" jsonschema:"description=Response from the router agent,required"`
	Error           string    `json:"error" jsonschema:"description=Error message if something went wrong,required"`
}

type Thought struct {
	Root     string   `json:"root" jsonschema:"description=The root thought,required"`
	Branches []string `json:"branches" jsonschema:"description=The branches of the thought,required"`
}

func NewUI() *UI {
	return &UI{}
}

func (ui *UI) Name() string {
	return "ui"
}

func (ui *UI) Description() string {
	return "The UI process is responsible for displaying the user prompt and response"
}

func (ui *UI) Schema() interface{} {
	return utils.GenerateSchema[UI]()
}
