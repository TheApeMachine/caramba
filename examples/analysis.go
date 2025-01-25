package examples

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
)

func RunAnalysis(ctx context.Context, prompt string) error {
	// Configure the task
	config := &ai.TaskConfig{
		InitialPrompt: prompt,
		ValidationRules: []ai.ValidationRule{
			{
				Name: "non-empty",
				Validator: func(i interface{}) (interface{}, error) {
					if i == nil {
						return nil, fmt.Errorf("response cannot be nil")
					}
					return i, nil
				},
			},
		},
		ConsensusConfig: drknow.ConsensusConfig{
			CollapseThreshold: 0.6,
			MinPerspectives:   2,
			Rules: []drknow.ConsensusRule{
				{
					Name:   "majority-vote",
					Weight: 0.6,
					Evaluate: func(perspectives []drknow.Perspective) (interface{}, float64) {
						if len(perspectives) == 0 {
							return nil, 0
						}

						// Count frequency of each answer
						counts := make(map[string]int)
						for _, p := range perspectives {
							if content, ok := p.Content.(string); ok {
								counts[content]++
							}
						}

						// Find most common answer
						var maxCount int
						var mostCommon string
						for answer, count := range counts {
							if count > maxCount {
								maxCount = count
								mostCommon = answer
							}
						}

						confidence := float64(maxCount) / float64(len(perspectives))
						return mostCommon, confidence
					},
				},
				{
					Name:   "latest-response",
					Weight: 0.4,
					Evaluate: func(perspectives []drknow.Perspective) (interface{}, float64) {
						if len(perspectives) == 0 {
							return nil, 0
						}
						latest := perspectives[len(perspectives)-1]
						// Give more weight to solver's response
						confidence := latest.Confidence
						if latest.Owner == "solver" {
							confidence *= 1.2 // Boost solver confidence by 20%
						}
						return latest.Content, confidence
					},
				},
			},
		},
		RequiredAgents: []string{
			"prompt_engineer",
			"reasoner",
			"challenger",
			"solver",
		},
		MaxIterations: 4,
	}

	// Create orchestrator
	orchestrator := ai.NewOrchestrator(ctx, config)

	// Create and register the required agents
	orchestrator.Agents = make(map[string][]*ai.Agent)
	orchestrator.Agents["prompt_engineer"] = []*ai.Agent{ai.NewAgent(ctx, "prompt_engineer", 1)}
	orchestrator.Agents["reasoner"] = []*ai.Agent{ai.NewAgent(ctx, "reasoner", 2)}
	orchestrator.Agents["challenger"] = []*ai.Agent{ai.NewAgent(ctx, "challenger", 2)}
	orchestrator.Agents["solver"] = []*ai.Agent{ai.NewAgent(ctx, "solver", 2)}

	// Create process
	process := ai.NewAnalysisProcess(ctx, prompt, config.MaxIterations)

	// Run the process
	if err := orchestrator.RunProcess(process); err != nil {
		return fmt.Errorf("failed to run analysis: %w", err)
	}

	return nil
}
