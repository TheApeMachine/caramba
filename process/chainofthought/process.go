package chainofthought

import "github.com/theapemachine/caramba/utils"

type Process struct {
	Thoughts    []Thought `json:"thoughts" jsonschema:"title=Thoughts,description=Your thoughts,required"`
	FinalAnswer string    `json:"final_answer" jsonschema:"title=FinalAnswer,description=The final answer that forms your response,required"`
}

type Thought struct {
	RootThought string    `json:"root_thought" jsonschema:"title=RootThought,description=A root thought from which other thoughts branch off,required"`
	Branches    []Thought `json:"branches" jsonschema:"title=Branches,description=A branch of thoughts related to the root thought,required"`
	Type        string    `json:"type" jsonschema:"title=Type,description=An identifier for the type of thought,required"`
}

func (cot *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
