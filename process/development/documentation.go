package development

import "github.com/theapemachine/amsh/utils"

type Documentation struct {
	DocTypes   []string `json:"doc_types" jsonschema:"description=Types of documentation to be generated, e.g., API docs, user guides"`
	StyleGuide string   `json:"style_guide" jsonschema:"description=Guidelines for formatting and style of documentation"`
}

func (documentation *Documentation) GenerateSchema() string {
	return utils.GenerateSchema[Documentation]()
}
