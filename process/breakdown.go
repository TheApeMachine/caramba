package process

import (
	"github.com/theapemachine/amsh/utils"
)

type Breakdown struct {
	Goal         string        `json:"goal" jsonschema:"title=Goal,description=The goal of the task,required"`
	Breakdown    string        `json:"breakdown" jsonschema:"title=Breakdown,description=The breakdown of the task,required"`
	Requirements []Requirement `json:"requirements" jsonschema:"title=Requirements,description=The requirements of the task,required"`
}

func (ta *Breakdown) SystemPrompt(key string) string {
	return utils.SystemPrompt(key, "breakdown", utils.GenerateSchema[Breakdown]())
}
