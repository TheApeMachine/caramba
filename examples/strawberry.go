package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/qpool"
)

func RunStrawberry() {
	ctx := context.Background()

	// Initialize agents
	promptAgent := ai.NewAgent(ctx, "prompt", 1)
	reasonerAgent := ai.NewAgent(ctx, "reasoner", 2)
	challengerAgent := ai.NewAgent(ctx, "challenger", 2)
	solverAgent := ai.NewAgent(ctx, "solver", 2)

	promptAgent.Initialize()
	reasonerAgent.Initialize()
	challengerAgent.Initialize()
	solverAgent.Initialize()

	// Configure and create the worker pool
	config := &qpool.Config{
		SchedulingTimeout: time.Second * 60,
	}
	pool := qpool.NewQ(ctx, 2, 4, config)
	defer pool.Close()

	// Create a broadcast group for events
	broadcast := pool.CreateBroadcastGroup("strawberry-events", time.Minute)
	events := pool.Subscribe("strawberry-events")

	// Create and configure the conversation thread
	message := provider.NewMessage(provider.RoleUser, `How many times do we find the letter r in the word strawberry?`)
	thread := provider.NewThread()
	thread.AddMessage(message)

	// Schedule prompt processing
	promptResult := pool.Schedule("prompt",
		func() (any, error) {
			for event := range promptAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("prompt", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule reasoning after prompt
	reasonerResult := pool.Schedule("reasoner",
		func() (any, error) {
			if err := (<-promptResult).Error; err != nil {
				return nil, err
			}
			for event := range reasonerAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("reasoner", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule challenger after reasoner
	challengerResult := pool.Schedule("challenger",
		func() (any, error) {
			if err := (<-reasonerResult).Error; err != nil {
				return nil, err
			}
			for event := range challengerAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("challenger", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule solver after challenger
	solverResult := pool.Schedule("solver",
		func() (any, error) {
			if err := (<-challengerResult).Error; err != nil {
				return nil, err
			}
			for event := range solverAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("solver", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	fmt.Println("\n=== Agent Responses ===")

	// Handle events from the broadcast group
	done := make(chan struct{})
	go func() {
		defer close(done)
		for value := range events {
			if event, ok := value.Value.(provider.Event); ok {
				switch event.Type {
				case provider.EventChunk:
					if event.Text != "" {
						fmt.Print(event.Text)
					}
				case provider.EventToolCall:
					fmt.Printf("\n🛠  Using tool: %s\n", event.Name)
				case provider.EventError:
					fmt.Printf("\n❌ Error: %s\n", event.Error)
				case provider.EventDone:
					fmt.Println("\n✅ Processing complete")
				}
			}
		}
	}()

	// Wait for final result or timeout
	select {
	case result := <-solverResult:
		if result.Error != nil {
			fmt.Printf("\n❌ Processing failed: %v\n", result.Error)
		} else {
			fmt.Println("\n=== Processing Complete ===")
		}
	case <-ctx.Done():
		fmt.Printf("\n⚠️  Processing timed out: %v\n", ctx.Err())
	}
}
