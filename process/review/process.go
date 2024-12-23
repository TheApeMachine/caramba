package review

import "github.com/theapemachine/amsh/utils"

type Process struct {
	Assessments []Assessment `json:"assessments" jsonschema:"title=Assessments,description=Your objective assessments of the agent's performance,required"`
	Performance string       `json:"performance" jsonschema:"title=Performance,description=Performance of the agent,enum=insufficient,enum=sufficient,enum=excellent,required"`
}

type Assessment struct {
	Question string `json:"question" jsonschema:"title=Question,description=A evaluation question that has a provable yes or no answer present in the response,required"`
	Answer   string `json:"answer" jsonschema:"title=Answer,description=The answer to the question,enum=yes,enum=no,required"`
	Fragment string `json:"fragment" jsonschema:"title=Fragment,description=The fragment of the response that supports the answer,required"`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
