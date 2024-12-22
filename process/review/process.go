package review

import "github.com/theapemachine/amsh/utils"

type Process struct {
	Assessment      []Assessment   `json:"assessment" jsonschema:"title=Assessment,description=The assessment of the Agent's layering process response,required"`
	FinalScores     []Score        `json:"final_scores" jsonschema:"title=Final Scores,description=Evaluation of different aspects of the response,required"`
	AggregatedScore float64        `json:"aggregated_score" jsonschema:"title=Aggregated Score,description=Overall quality score for potential training example,required"`
	Optimizations   []Optimization `json:"optimizations" jsonschema:"title=Optimizations,description=Suggestions for improvement and training data formatting,required"`
}

type Assessment struct {
	Category         string   `json:"category" jsonschema:"title=Category,description=Aspect being assessed,enum=parallel_processing,enum=workload_combinations,enum=specialized_processes,enum=conceptual_structure,required"`
	MessageFragments []string `json:"message_fragments" jsonschema:"title=Message Fragments,description=Relevant parts of the response demonstrating quality or issues,required"`
	Score            float64  `json:"score" jsonschema:"title=Score,description=Score for this aspect (0-1),required"`
	Reasoning        string   `json:"reasoning" jsonschema:"title=Reasoning,description=Explanation of the score and assessment,required"`
}

type Score struct {
	Category    string  `json:"category" jsonschema:"title=Category,description=Aspect being scored,enum=parallel_processing,enum=workload_combinations,enum=specialized_processes,enum=conceptual_structure,required"`
	Score       float64 `json:"score" jsonschema:"title=Score,description=Final score for this category,required"`
	IsTrainable bool    `json:"is_trainable" jsonschema:"title=Is Trainable,description=Whether this aspect is good enough for training data,required"`
}

type Optimization struct {
	Type           string  `json:"type" jsonschema:"title=Type,description=Type of optimization,enum=parameter,enum=training_format,enum=improvement,required"`
	Parameter      string  `json:"parameter,omitempty" jsonschema:"title=Parameter,description=Parameter to adjust if type is parameter"`
	NewValue       float64 `json:"new_value,omitempty" jsonschema:"title=New Value,description=New parameter value if type is parameter"`
	TrainingFormat string  `json:"training_format,omitempty" jsonschema:"title=Training Format,description=Formatted training example if quality is sufficient"`
	Suggestion     string  `json:"suggestion,omitempty" jsonschema:"title=Suggestion,description=Specific improvement suggestion if needed"`
}

func (p *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
