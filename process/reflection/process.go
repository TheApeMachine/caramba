package reflection

import "github.com/theapemachine/caramba/utils"

// ReflectionPoint represents a specific aspect of the response to reflect upon
type ReflectionPoint struct {
	Aspect          string   `json:"aspect" jsonschema:"title=Aspect,description=The specific aspect being reflected upon (e.g., 'assumptions', 'evidence', 'bias'),required"`
	Critique        string   `json:"critique" jsonschema:"title=Critique,description=Critical analysis of this aspect,required"`
	Improvements    []string `json:"improvements" jsonschema:"title=Improvements,description=Specific ways to improve this aspect"`
	ConfidenceLevel int      `json:"confidence_level" jsonschema:"title=ConfidenceLevel,description=Confidence in the original response"`
}

type Process struct {
	PreviousResponse string            `json:"previous_response" jsonschema:"title=PreviousResponse,description=The response being reflected upon,required"`
	ReflectionPoints []ReflectionPoint `json:"reflection_points" jsonschema:"title=ReflectionPoints,description=Different aspects of self-reflection"`
	RevisionNeeded   bool              `json:"revision_needed" jsonschema:"title=RevisionNeeded,description=Whether the previous response needs significant revision"`
	Conclusion       string            `json:"conclusion" jsonschema:"title=Conclusion,description=Final reflection summary and action items"`
}

func (p *Process) Name() string {
	return "Reflection"
}

func (p *Process) Description() string {
	return "A metacognitive process that critically examines previous responses and thoughts, identifying potential biases, assumptions, and areas for improvement."
}

func (p *Process) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Process]()
}
