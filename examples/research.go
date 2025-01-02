package examples

import (
	"context"
	"fmt"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/qpool"
)

func RunResearch() {
	ctx := context.Background()

	inputAgent := ai.NewAgent(ctx, "prompt", 1)
	searchAgent := ai.NewAgent(ctx, "researcher", 2)
	analyzeAgent := ai.NewAgent(ctx, "analyst", 2)
	writeAgent := ai.NewAgent(ctx, "writer", 1)

	searchAgent.AddTools(
		tools.NewBrowser(),
		tools.NewQdrantStore("research", 1536),
	)
	analyzeAgent.AddTools(
		tools.NewQdrantQuery("research", 1536),
	)

	inputAgent.Initialize()
	searchAgent.Initialize()
	analyzeAgent.Initialize()
	writeAgent.Initialize()

	// Configure and create the worker pool
	config := &qpool.Config{
		SchedulingTimeout: time.Second * 60,
	}
	pool := qpool.NewQ(ctx, 2, 4, config)
	defer pool.Close()

	// Create a broadcast group for events
	broadcast := pool.CreateBroadcastGroup("research-events", time.Minute)
	events := pool.Subscribe("research-events")

	fmt.Println("🔍 Research Pipeline Example")
	fmt.Println("Enter your research question:")

	var question string
	fmt.Scanln(&question)

	message := provider.NewMessage(provider.RoleUser, question)
	thread := provider.NewThread()
	thread.AddMessage(message)

	// Schedule input processing
	inputResult := pool.Schedule("input",
		func() (any, error) {
			for event := range inputAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("input", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule search after input
	searchResult := pool.Schedule("search",
		func() (any, error) {
			if err := (<-inputResult).Error; err != nil {
				return nil, err
			}
			for event := range searchAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("search", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule analysis after search
	analyzeResult := pool.Schedule("analyze",
		func() (any, error) {
			if err := (<-searchResult).Error; err != nil {
				return nil, err
			}
			for event := range analyzeAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("analyze", 3, time.Minute),
		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
		qpool.WithTTL(time.Minute),
	)

	// Schedule writing after analysis
	writeResult := pool.Schedule("write",
		func() (any, error) {
			if err := (<-analyzeResult).Error; err != nil {
				return nil, err
			}
			for event := range writeAgent.Generate(ctx, message) {
				broadcast.Send(qpool.QuantumValue{Value: event})
			}
			return nil, nil
		},
		qpool.WithCircuitBreaker("write", 3, time.Minute),
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
					fmt.Println("\n✅ Research complete")
				}
			}
		}
	}()

	// Wait for final result or timeout
	select {
	case result := <-writeResult:
		if result.Error != nil {
			fmt.Printf("\n❌ Research failed: %v\n", result.Error)
		} else {
			fmt.Println("\n=== Research Complete ===")
		}
	case <-ctx.Done():
		fmt.Printf("\n⚠️  Research timed out: %v\n", ctx.Err())
	}
}
