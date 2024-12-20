package development

import "github.com/theapemachine/amsh/utils"

type Implementation struct {
	CodeStandards string   `json:"code_standards" jsonschema:"description=Guidelines and standards for writing code"`
	Technologies  []string `json:"technologies" jsonschema:"description=Programming languages, frameworks, or libraries used in the project"`
	Modules       []Module `json:"modules" jsonschema:"description=Modules or features to be implemented"`
}

type Module struct {
	Name        string   `json:"name" jsonschema:"description=Module or feature name"`
	Description string   `json:"description" jsonschema:"description=Brief description of the module's purpose"`
	Tasks       []string `json:"tasks" jsonschema:"description=List of tasks for implementing the module"`
}

func (implementation *Implementation) GenerateSchema() string {
	return utils.GenerateSchema[Implementation]()
}
