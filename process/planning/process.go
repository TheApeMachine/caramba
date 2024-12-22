package planning

import "github.com/theapemachine/amsh/utils"

type Process struct {
	Epics []Epic `json:"epics" jsonschema:"title=Epics,description=The epics that are needed to achieve the goal,required"`
}

type Epic struct {
	Title       string  `json:"title" jsonschema:"title=Title,description=The title of the epic,required"`
	Description string  `json:"description" jsonschema:"title=Description,description=The description of the epic,required"`
	Stories     []Story `json:"stories" jsonschema:"title=Stories,description=The stories that are needed to achieve the epic written in Gherkin,required"`
}

type Story struct {
	Title       string `json:"title" jsonschema:"title=Title,description=The title of the story,required"`
	Description string `json:"description" jsonschema:"title=Description,description=The description of the story written in Gherkin,required"`
	Tasks       []Task `json:"tasks" jsonschema:"title=Tasks,description=The tasks that are needed to achieve the story,required"`
}

type Task struct {
	Title       string `json:"title" jsonschema:"title=Title,description=The title of the task,required"`
	Description string `json:"description" jsonschema:"title=Description,description=The description of the task written in Gherkin,required"`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
