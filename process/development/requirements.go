package development

import "github.com/theapemachine/amsh/utils"

type Requirements struct {
	Functional    []Requirement `json:"functional" jsonschema:"description=List of functional requirements defining specific system behaviors"`
	NonFunctional []Requirement `json:"non_functional" jsonschema:"description=List of non-functional requirements like performance, scalability, and security"`
}

type Requirement struct {
	Description string `json:"description" jsonschema:"description=A description of the requirement"`
	Priority    string `json:"priority" jsonschema:"description=Priority level (e.g., high, medium, low)"`
}

func (requirements *Requirements) GenerateSchema() string {
	return utils.GenerateSchema[Requirements]()
}
