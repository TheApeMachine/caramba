package ui

import "github.com/theapemachine/caramba/utils"

type Process struct {
	RelevantFragments []string          `json:"relevant_fragments" jsonschema:"title=Relevant Fragments,description=The most relevant fragments found in the context,required"`
	CandidateAnswers  []CandidateAnswer `json:"candidate_answers" jsonschema:"title=Candidate Answers,description=The candidate answers to the user's prompt,required"`
	NeedsIteration    bool              `json:"needs_iteration" jsonschema:"title=Needs Iteration,description=Whether the user's prompt needs to be iterated on,required"`
	FinalAnswer       string            `json:"final_answer" jsonschema:"title=Final Answer,description=The final answer to the user's prompt"`
}

type CandidateAnswer struct {
	Answer      string  `json:"answer" jsonschema:"title=Answer,description=The candidate answer to the user's prompt"`
	Quality     float64 `json:"quality" jsonschema:"title=Quality,description=The quality of the candidate answer,enum=poor,enum=fair,enum=good,enum=excellent,required"`
	Explanation string  `json:"explanation" jsonschema:"title=Explanation,description=The explanation of the candidate answer"`
}

func (process *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
