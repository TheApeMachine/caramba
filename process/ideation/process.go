package ideation

import "github.com/theapemachine/amsh/utils"

/*
Process represents the ideation process.
*/
type Process struct {
	Moonshot Moonshot `json:"moonshot" jsonschema:"required,..."`
	Sensible Sensible `json:"sensible" jsonschema:"required,..."`
	Catalyst Catalyst `json:"catalyst" jsonschema:"required,..."`
	Guardian Guardian `json:"guardian" jsonschema:"required,..."`
}

type Moonshot struct {
	Ideas []Idea `json:"ideas" jsonschema:"required,title=Moonshot Ideas,description=A list of moonshot ideas."`
}

type Sensible struct {
	Ideas []Idea `json:"ideas" jsonschema:"required,title=Sensible Ideas,description=A list of sensible ideas."`
}

type Catalyst struct {
	Ideas []Idea `json:"ideas" jsonschema:"required,title=Catalyst Ideas,description=A list of catalyst ideas."`
}

type Guardian struct {
	Ideas []Idea `json:"ideas" jsonschema:"required,title=Guardian Ideas,description=A list of guardian ideas."`
}

type Idea struct {
	Pitch      string   `json:"pitch" jsonschema:"required,title=Pitch,description=A short pitch for the idea."`
	Motivation string   `json:"motivation" jsonschema:"required,title=Motivation,description=The motivation for the idea."`
	Arguments  []string `json:"arguments" jsonschema:"required,title=Arguments,description=The arguments for the idea."`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
