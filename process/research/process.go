package research

import "github.com/theapemachine/amsh/utils"

/*
Process defines the process for conducting research on a specific topic.
*/
type Process struct {
	Topic string   `json:"topic" jsonschema:"title=Topic,description=The topic of the research,required"`
	Notes string   `json:"notes" jsonschema:"title=Notes,description=Additional notes or context for the research"`
	EOIs  []string `json:"eois" jsonschema:"title=Entities of Interest,description=The Entities of Interest list specifically significant entities which are relevant to the research"`
}

func (process *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
