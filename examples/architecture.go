package examples

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/qpool"
)

/*
Architecture is an experiment to have the agents come up with a new AI architecture.
This tests the model's ability to think creatively, but also practically.
*/
type Architecture struct {
	ctx    context.Context
	agents map[string][]*ai.Agent
	pool   *qpool.Q
}

/*
NewArchitecture creates a new Architecture instance with the specified context and role.
It initializes an empty scratchpad for accumulating assistant responses
and sets up the basic formatting configuration.

Parameters:
  - ctx: The context for operations
  - role: The role designation for the AI agent
*/
func NewArchitecture(ctx context.Context, role string) *Architecture {
	return &Architecture{
		ctx: ctx,
		agents: map[string][]*ai.Agent{
			"prompt_engineer": {ai.NewAgent(ctx, "prompt_engineer", 1)},
			"reasoner":        {ai.NewAgent(ctx, "reasoner", 2)},
			"challenger":      {ai.NewAgent(ctx, "challenger", 2)},
			"solver":          {ai.NewAgent(ctx, "solver", 2)},
		},
		pool: qpool.NewQ(ctx, 4, 8, &qpool.Config{
			SchedulingTimeout: time.Second * 60,
		}),
	}
}

/*
Run starts the architecture loop, which allows the user to interact with the agent.
It initializes the agents and creates a worker pool to handle the different
steps of the architecture problem.
*/
func (s *Architecture) Run() error {
	var wg sync.WaitGroup
	wg.Add(1)

	// Create consensus space for the architecture problem
	consensus := drknow.NewConsensusSpace("architecture-space", drknow.ConsensusConfig{
		CollapseThreshold: 0.8,
		MinPerspectives:   2,
		Rules: []drknow.ConsensusRule{
			{
				Name:   "majority-vote",
				Weight: 0.6,
				Evaluate: func(perspectives []drknow.Perspective) (interface{}, float64) {
					// Count frequency of each answer
					counts := make(map[interface{}]int)
					for _, p := range perspectives {
						counts[p.Content]++
					}

					// Find most common answer and calculate confidence
					var maxCount int
					var mostCommon interface{}
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
				Name:   "confidence-weighted",
				Weight: 0.4,
				Evaluate: func(perspectives []drknow.Perspective) (interface{}, float64) {
					// Weight answers by agent confidence
					weighted := make(map[interface{}]float64)
					for _, p := range perspectives {
						weighted[p.Content] += p.Confidence
					}

					// Find highest confidence answer
					var maxConfidence float64
					var bestAnswer interface{}
					for answer, confidence := range weighted {
						if confidence > maxConfidence {
							maxConfidence = confidence
							bestAnswer = answer
						}
					}

					return bestAnswer, maxConfidence / float64(len(perspectives))
				},
			},
		},
	})

	// Set up dependencies between agents
	consensus.AddDependency("reasoner", []string{"prompt_engineer"})
	consensus.AddDependency("challenger", []string{"reasoner"})
	consensus.AddDependency("solver", []string{"challenger", "reasoner"})

	// Handle consensus collapse
	consensus.OnCollapse = func(result interface{}) {
		count, ok := result.(int)
		if ok {
			s.pool.CreateBroadcastGroup("final-answer", time.Minute).
				Send(qpool.NewQValue(count, nil))
		}
		wg.Done()
	}

	// Schedule the agent interactions
	s.pool.Schedule("prompt_engineer",
		func() (any, error) {
			// Start with prompt agent
			promptChan := s.agents["prompt_engineer"][0].Generate(s.ctx, &provider.Message{
				Role:    "user",
				Content: "Come up with an entirely new AI architecture, that is not based on anything we've seen before. It should be a new way of thinking about AI, and it should be practical enough to be able to run on consumer hardware. Provide a detailed plan for how to implement it, and a detailed plan for how to train it.",
			})

			accumulator := stream.NewAccumulator()
			for event := range accumulator.Generate(s.ctx, promptChan) {
				fmt.Print(event.Text)
			}

			perspective := drknow.Perspective{
				ID:         "prompt_engineer",
				Owner:      "prompt_engineer",
				Content:    accumulator.Compile().Text,
				Confidence: 0.6, // We'll need to derive this from the response
				Method:     "llm-generation",
				Reasoning:  []string{accumulator.Compile().Text},
				Timestamp:  time.Now(),
			}
			consensus.AddPerspective(perspective)

			return nil, nil
		},
	)

	s.pool.Schedule("reasoner",
		func() (any, error) {
			// Only proceed with reasoner if prompt is done
			if consensus.HasPerspective("prompt_engineer") {
				reasonerChan := s.agents["reasoner"][0].Generate(s.ctx, &provider.Message{
					Role:    "user",
					Content: "Verify the answer provided by the previous agent by presenting your own empirical evidence.",
				})

				accumulator := stream.NewAccumulator()
				for event := range accumulator.Generate(s.ctx, reasonerChan) {
					fmt.Print(event.Text)
				}

				perspective := drknow.Perspective{
					ID:         "reasoner",
					Owner:      "reasoner",
					Content:    accumulator.Compile().Text,
					Confidence: 0.9, // Higher confidence for systematic analysis
					Method:     "systematic-analysis",
					Reasoning:  []string{accumulator.Compile().Text},
					Timestamp:  time.Now(),
				}

				consensus.AddPerspective(perspective)
			}

			return nil, nil
		},
		qpool.WithDependencies([]string{"prompt_engineer"}),
		qpool.WithDependencyRetry(3, &qpool.ExponentialBackoff{
			Initial: time.Second * 10,
		}),
	)

	s.pool.Schedule("challenger",
		func() (any, error) {
			// Only proceed with challenger if reasoner is done
			if consensus.HasPerspective("reasoner") {
				challengerChan := s.agents["challenger"][0].Generate(s.ctx, &provider.Message{
					Role:    "user",
					Content: "Challenge the work of the previous agent by presenting alternative angles, counter-examples, or edge cases.",
				})

				accumulator := stream.NewAccumulator()

				for event := range accumulator.Generate(s.ctx, challengerChan) {
					fmt.Print(event.Text)
				}

				perspective := drknow.Perspective{
					ID:         "challenger",
					Owner:      "challenger",
					Content:    accumulator.Compile().Text,
					Confidence: 0.95, // High confidence for verification
					Method:     "verification",
					Reasoning:  []string{accumulator.Compile().Text},
					Timestamp:  time.Now(),
				}

				consensus.AddPerspective(perspective)
			}

			return nil, nil
		},
		qpool.WithDependencies([]string{"reasoner"}),
		qpool.WithDependencyRetry(3, &qpool.ExponentialBackoff{
			Initial: time.Second * 10,
		}),
	)

	s.pool.Schedule("solver",
		func() (any, error) {
			// Only proceed with solver if both challenger and reasoner are done
			if consensus.HasPerspective("challenger") && consensus.HasPerspective("reasoner") {
				solverChan := s.agents["solver"][0].Generate(s.ctx, &provider.Message{
					Role:    "user",
					Content: "Consider the work of the previous agents and motivate the final answer, and the confidence in that answer.",
				})

				accumulator := stream.NewAccumulator()
				for event := range accumulator.Generate(s.ctx, solverChan) {
					fmt.Print(event.Text)
				}

				perspective := drknow.Perspective{
					ID:         "solver",
					Owner:      "solver",
					Content:    accumulator.Compile().Text,
					Confidence: 1.0, // Maximum confidence for final solution
					Method:     "final-verification",
					Reasoning:  []string{accumulator.Compile().Text},
					Timestamp:  time.Now(),
				}
				consensus.AddPerspective(perspective)
			}

			return nil, nil
		},
		qpool.WithDependencies([]string{"challenger", "reasoner"}),
		qpool.WithDependencyRetry(3, &qpool.ExponentialBackoff{
			Initial: time.Second * 10,
		}),
	)

	// Wait for consensus to be reached
	wg.Wait()
	return nil
}
