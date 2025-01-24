package ai

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/provider"
)

type AnalysisProcess struct {
	ctx           context.Context
	initialPrompt string
	state         map[string]interface{}
	iterations    int
	maxIterations int
}

func NewAnalysisProcess(ctx context.Context, prompt string, maxIterations int) *AnalysisProcess {
	return &AnalysisProcess{
		ctx:           ctx,
		initialPrompt: prompt,
		state:         make(map[string]interface{}),
		maxIterations: maxIterations,
	}
}

func (p *AnalysisProcess) Initialize(ctx context.Context) error {
	p.ctx = ctx
	return nil
}

func (p *AnalysisProcess) GeneratePrompt(role string, state interface{}) (*provider.Message, error) {
	var content string

	switch role {
	case "prompt_engineer":
		content = p.initialPrompt
	case "reasoner":
		content = fmt.Sprintf("Analyze the following response and provide your perspective:\n\n%v",
			p.state["prompt_engineer_response"])
	case "challenger":
		content = fmt.Sprintf("Challenge the following analyses:\n\nInitial: %v\n\nReasoner: %v",
			p.state["prompt_engineer_response"],
			p.state["reasoner_response"])
	case "solver":
		content = fmt.Sprintf("Synthesize a final answer from these perspectives:\n\n%v\n\n%v\n\n%v",
			p.state["prompt_engineer_response"],
			p.state["reasoner_response"],
			p.state["challenger_response"])
	default:
		return nil, fmt.Errorf("unknown role: %s", role)
	}

	return &provider.Message{
		Role:    "user",
		Content: content,
	}, nil
}

func (p *AnalysisProcess) ValidateResponse(response interface{}) (interface{}, error) {
	// Add validation logic specific to your use case
	if response == nil {
		return nil, fmt.Errorf("response cannot be nil")
	}
	return response, nil
}

func (p *AnalysisProcess) UpdateState(agentID string, response interface{}) error {
	p.state[fmt.Sprintf("%s_response", agentID)] = response
	p.iterations++
	return nil
}

func (p *AnalysisProcess) IsComplete() bool {
	return p.iterations >= p.maxIterations || len(p.state) >= 4 // All agents have responded
}
