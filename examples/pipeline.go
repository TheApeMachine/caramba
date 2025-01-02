package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/qpool"
)

func RunPipeline() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// Initialize agents with specific roles
	analyzeAgent := ai.NewAgent(ctx, "analyzer", 2)
	researcher1 := ai.NewAgent(ctx, "researcher", 2)
	researcher2 := ai.NewAgent(ctx, "researcher", 2)
	writerAgent := ai.NewAgent(ctx, "writer", 1)

	// Configure and create the worker pool
	pool := qpool.NewQ(ctx, 2, 4, &qpool.Config{
		SchedulingTimeout: time.Second * 60,
	})

	defer pool.Close()

	// Create a broadcast group for events
	broadcast := pool.CreateBroadcastGroup("pipeline-events", time.Minute)
	events := pool.Subscribe("pipeline-events")

	// Initial message
	message := provider.NewMessage(provider.RoleUser, "Explain how quantum computing works")

	fmt.Println("🔄 Pipeline Example")
	fmt.Println("=== Processing Pipeline ===")
	fmt.Println("1. Analyzing input...")

	// Schedule analyzer job
	analyzeResult := pool.Schedule("analyze",
		func() (any, error) {
			for event := range analyzeAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("analyzer", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule researcher jobs
	research1Result := pool.Schedule("research1",
		func() (any, error) {
			if err := (<-analyzeResult).Error; err != nil {
				return nil, err
			}
			for event := range researcher1.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("research1", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	research2Result := pool.Schedule("research2",
		func() (any, error) {
			if err := (<-analyzeResult).Error; err != nil {
				return nil, err
			}
			for event := range researcher2.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("research2", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule writer job
	writerResult := pool.Schedule("write",
		func() (any, error) {
			// Wait for all results
			if err := (<-research1Result).Error; err != nil {
				return nil, err
			}
			if err := (<-research2Result).Error; err != nil {
				return nil, err
			}

			for event := range writerAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("writer", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

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
	case result := <-writerResult:
		if result.Error != nil {
			fmt.Printf("\n❌ Pipeline failed: %v\n", result.Error)
		} else {
			fmt.Println("\n=== Pipeline Complete ===")
		}
	case <-ctx.Done():
		fmt.Printf("\n⚠️  Processing timed out: %v\n", ctx.Err())
	}
}
