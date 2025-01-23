package examples

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/qpool"
)

type Strawberry struct {
	ctx    context.Context
	agents map[string][]*ai.Agent
	pool   *qpool.Q
}

func NewStrawberry(ctx context.Context, role string) *Strawberry {
	return &Strawberry{
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

func (s *Strawberry) Run() error {
	var wg sync.WaitGroup
	wg.Add(1)

	// Create consensus space
	consensus := drknow.NewConsensusSpace("strawberry-count", drknow.ConsensusConfig{
		CollapseThreshold: 0.8,
		MinPerspectives:   2,
		Rules: []drknow.ConsensusRule{
			{
				Name:   "majority-vote",
				Weight: 0.6,
				Evaluate: func(perspectives []drknow.Perspective) (interface{}, float64) {
					counts := make(map[interface{}]int)
					for _, p := range perspectives {
						counts[p.Content]++
					}

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
					weighted := make(map[interface{}]float64)
					for _, p := range perspectives {
						weighted[p.Content] += p.Confidence
					}

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

	// Create entanglement between all agents
	entanglement := qpool.NewEntanglement("strawberry-analysis", []qpool.Job{}, time.Hour)

	// Set up dependencies
	consensus.AddDependency("reasoner", []string{"prompt_engineer"})
	consensus.AddDependency("challenger", []string{"reasoner"})
	consensus.AddDependency("solver", []string{"challenger", "reasoner"})

	// Handle consensus collapse
	consensus.OnCollapse = func(result interface{}) {
		count, ok := result.(int)
		if ok {
			s.pool.CreateBroadcastGroup("final-answer", time.Minute).
				Send(qpool.NewQValue(count, []qpool.State{
					{Value: count, Probability: 1.0},
				}))
		}
		wg.Done()
	}

	// Schedule the prompt engineer
	s.pool.Schedule("prompt_engineer",
		func() (any, error) {
			promptChan := s.agents["prompt_engineer"][0].Generate(s.ctx, &provider.Message{
				Role:    "user",
				Content: "How many times do we find the letter r in the word strawberry?",
			})

			accumulator := stream.NewAccumulator()
			for event := range accumulator.Generate(s.ctx, promptChan) {
				if data, ok := event.Data().(map[string]interface{}); ok {
					if text, ok := data["text"].(string); ok {
						fmt.Print(text)
					}
				}
			}
			response := accumulator.Compile()
			data := response.Data().(map[string]interface{})
			text, _ := data["text"].(string)

			// Update shared state
			entanglement.UpdateState("prompt_engineer_response", text)
			entanglement.UpdateState("prompt_engineer_count", s.extractCount(text))
			entanglement.UpdateState("prompt_engineer_confidence", 0.6)

			perspective := drknow.Perspective{
				ID:         "prompt_engineer",
				Owner:      "prompt_engineer",
				Content:    text,
				Confidence: 0.6,
				Method:     "llm-generation",
				Reasoning:  []string{text},
				Timestamp:  time.Now(),
			}
			consensus.AddPerspective(perspective)

			return nil, nil
		},
	)

	// Schedule the reasoner
	s.pool.Schedule("reasoner",
		func() (any, error) {
			// Get previous work from entanglement
			promptResponse, _ := entanglement.GetState("prompt_engineer_response")
			promptCount, _ := entanglement.GetState("prompt_engineer_count")

			// Build context-aware prompt
			prompt := fmt.Sprintf(`Previous agent's analysis:
%s

Their count was: %v

Please verify this answer by:
1. Actually counting the letter 'r' in 'strawberry'
2. Explaining your counting method
3. Comparing your result with the previous agent
4. Providing your confidence level`, promptResponse, promptCount)

			reasonerChan := s.agents["reasoner"][0].Generate(s.ctx, &provider.Message{
				Role:    "user",
				Content: prompt,
			})

			accumulator := stream.NewAccumulator()
			for event := range accumulator.Generate(s.ctx, reasonerChan) {
				if data, ok := event.Data().(map[string]interface{}); ok {
					if text, ok := data["text"].(string); ok {
						fmt.Print(text)
					}
				}
			}
			response := accumulator.Compile()
			data := response.Data().(map[string]interface{})
			text, _ := data["text"].(string)

			// Update shared state
			entanglement.UpdateState("reasoner_response", text)
			entanglement.UpdateState("reasoner_count", s.extractCount(text))
			entanglement.UpdateState("reasoner_confidence", 0.9)

			perspective := drknow.Perspective{
				ID:         "reasoner",
				Owner:      "reasoner",
				Content:    text,
				Confidence: 0.9,
				Method:     "systematic-analysis",
				Reasoning:  []string{text},
				Timestamp:  time.Now(),
			}
			consensus.AddPerspective(perspective)

			return nil, nil
		},
		qpool.WithDependencies([]string{"prompt_engineer"}),
		qpool.WithDependencyRetry(3, &qpool.ExponentialBackoff{Initial: time.Second * 10}),
	)

	// Schedule the challenger
	s.pool.Schedule("challenger",
		func() (any, error) {
			// Get previous work from entanglement
			promptResponse, _ := entanglement.GetState("prompt_engineer_response")
			reasonerResponse, _ := entanglement.GetState("reasoner_response")
			promptCount, _ := entanglement.GetState("prompt_engineer_count")
			reasonerCount, _ := entanglement.GetState("reasoner_count")

			// Build context-aware prompt
			prompt := fmt.Sprintf(`Previous analyses:

Prompt Engineer:
%s
Count: %v

Reasoner:
%s
Count: %v

Please challenge these analyses by:
1. Identifying any potential mistakes or oversights
2. Providing your own careful count
3. Explaining any discrepancies
4. Rating your confidence in your answer`, promptResponse, promptCount, reasonerResponse, reasonerCount)

			challengerChan := s.agents["challenger"][0].Generate(s.ctx, &provider.Message{
				Role:    "user",
				Content: prompt,
			})

			accumulator := stream.NewAccumulator()
			for event := range accumulator.Generate(s.ctx, challengerChan) {
				if data, ok := event.Data().(map[string]interface{}); ok {
					if text, ok := data["text"].(string); ok {
						fmt.Print(text)
					}
				}
			}
			response := accumulator.Compile()
			data := response.Data().(map[string]interface{})
			text, _ := data["text"].(string)

			// Update shared state
			entanglement.UpdateState("challenger_response", text)
			entanglement.UpdateState("challenger_count", s.extractCount(text))
			entanglement.UpdateState("challenger_confidence", 0.95)

			perspective := drknow.Perspective{
				ID:         "challenger",
				Owner:      "challenger",
				Content:    text,
				Confidence: 0.95,
				Method:     "verification",
				Reasoning:  []string{text},
				Timestamp:  time.Now(),
			}
			consensus.AddPerspective(perspective)

			return nil, nil
		},
		qpool.WithDependencies([]string{"reasoner"}),
		qpool.WithDependencyRetry(3, &qpool.ExponentialBackoff{Initial: time.Second * 10}),
	)

	// Schedule the solver
	s.pool.Schedule("solver",
		func() (any, error) {
			// Get full conversation history from entanglement
			promptResponse, _ := entanglement.GetState("prompt_engineer_response")
			reasonerResponse, _ := entanglement.GetState("reasoner_response")
			challengerResponse, _ := entanglement.GetState("challenger_response")
			promptCount, _ := entanglement.GetState("prompt_engineer_count")
			reasonerCount, _ := entanglement.GetState("reasoner_count")
			challengerCount, _ := entanglement.GetState("challenger_count")

			// Build context-aware prompt
			prompt := fmt.Sprintf(`Complete analysis history:

Prompt Engineer:
%s
Count: %v

Reasoner:
%s
Count: %v

Challenger:
%s
Count: %v

Please provide a final analysis:
1. Evaluate all previous counts and reasoning
2. Determine the correct count with explanation
3. Assess the confidence level of your conclusion
4. Explain any corrections needed to previous analyses`,
				promptResponse, promptCount,
				reasonerResponse, reasonerCount,
				challengerResponse, challengerCount)

			solverChan := s.agents["solver"][0].Generate(s.ctx, &provider.Message{
				Role:    "user",
				Content: prompt,
			})

			accumulator := stream.NewAccumulator()
			for event := range accumulator.Generate(s.ctx, solverChan) {
				if data, ok := event.Data().(map[string]interface{}); ok {
					if text, ok := data["text"].(string); ok {
						fmt.Print(text)
					}
				}
			}
			response := accumulator.Compile()
			data := response.Data().(map[string]interface{})
			text, _ := data["text"].(string)

			// Update shared state
			entanglement.UpdateState("solver_response", text)
			entanglement.UpdateState("solver_count", s.extractCount(text))
			entanglement.UpdateState("solver_confidence", 1.0)

			perspective := drknow.Perspective{
				ID:         "solver",
				Owner:      "solver",
				Content:    text,
				Confidence: 1.0,
				Method:     "final-verification",
				Reasoning:  []string{text},
				Timestamp:  time.Now(),
			}
			consensus.AddPerspective(perspective)

			return nil, nil
		},
		qpool.WithDependencies([]string{"challenger", "reasoner"}),
		qpool.WithDependencyRetry(3, &qpool.ExponentialBackoff{Initial: time.Second * 10}),
	)

	// Wait for consensus
	wg.Wait()
	return nil
}

func (s *Strawberry) extractCount(text string) int {
	var count int
	_, err := fmt.Sscanf(text, "%d", &count)
	if err != nil {
		// Try to parse number words
		text = strings.ToLower(text)
		if strings.Contains(text, "three") || strings.Contains(text, "3") {
			count = 3
		} else if strings.Contains(text, "two") || strings.Contains(text, "2") {
			count = 2
		} else if strings.Contains(text, "one") || strings.Contains(text, "1") {
			count = 1
		}
	}
	return count
}
