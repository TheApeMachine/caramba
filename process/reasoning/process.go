package reasoning

import "github.com/theapemachine/caramba/utils"

// ThoughtContent separates the content from the recursive structure
type ThoughtContent struct {
	RootThought string `json:"root_thought" jsonschema:"title=RootThought,description=A root thought from which other thoughts branch off,required"`
	Type        string `json:"type" jsonschema:"title=Type,description=An identifier for the type of thought,required"`
}

// Thought now references ThoughtContent to break the recursive cycle
type Thought struct {
	ThoughtContent
	Branches []ThoughtContent `json:"branches" jsonschema:"title=Branches,description=A branch of thoughts related to the root thought,required"`
}

type Process struct {
	Thoughts    []Thought `json:"thoughts" jsonschema:"title=Thoughts,description=Your thoughts,required"`
	FinalAnswer string    `json:"final_answer" jsonschema:"title=FinalAnswer,description=The final answer that forms your response which you can leave empty to iterate on your current thoughts"`
}

func (proc *Process) Name() string {
	return "Reasoning"
}

func (cot *Process) Description() string {
	return "A structured reasoning process that allows you to branch out across multiple viewpoints, designed for deep exploration of concepts."
}

func (cot *Process) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Process]()
}
