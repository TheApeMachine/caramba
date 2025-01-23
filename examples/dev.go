package examples

// func RunDev() {
// 	ctx := context.Background()

// 	designAgent := ai.NewAgent(ctx, "architect", 1)
// 	implementAgent := ai.NewAgent(ctx, "developer", 2)
// 	testAgent := ai.NewAgent(ctx, "tester", 2)
// 	reviewAgent := ai.NewAgent(ctx, "reviewer", 1)

// 	designAgent.AddTools(provider.NewToolset("architect")...)
// 	implementAgent.AddTools(provider.NewToolset("developer")...)
// 	testAgent.AddTools(provider.NewToolset("tester")...)
// 	reviewAgent.AddTools(provider.NewToolset("reviewer")...)

// 	designAgent.Initialize()
// 	implementAgent.Initialize()
// 	testAgent.Initialize()
// 	reviewAgent.Initialize()

// 	// Configure and create the worker pool
// 	config := &qpool.Config{
// 		SchedulingTimeout: time.Second * 60,
// 	}
// 	pool := qpool.NewQ(ctx, 2, 4, config)
// 	defer pool.Close()

// 	// Create a broadcast group for events
// 	broadcast := pool.CreateBroadcastGroup("dev-events", time.Minute)
// 	events := pool.Subscribe("dev-events")

// 	fmt.Println("👩‍💻 Development Pipeline Example")
// 	fmt.Println("Enter the feature requirements:")

// 	var requirements string
// 	fmt.Scanln(&requirements)

// 	message := provider.NewMessage(provider.RoleUser, requirements)
// 	thread := provider.NewThread()
// 	thread.AddMessage(message)

// 	// Schedule design phase
// 	designResult := pool.Schedule("design",
// 		func() (any, error) {
// 			for event := range designAgent.Generate(ctx, message) {
// 				broadcast.Send(qpool.QuantumValue{Value: event})
// 			}
// 			return nil, nil
// 		},
// 		qpool.WithCircuitBreaker("design", 3, time.Minute),
// 		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
// 		qpool.WithTTL(time.Minute),
// 	)

// 	// Schedule implementation after design
// 	implementResult := pool.Schedule("implement",
// 		func() (any, error) {
// 			if err := (<-designResult).Error; err != nil {
// 				return nil, err
// 			}
// 			for event := range implementAgent.Generate(ctx, message) {
// 				broadcast.Send(qpool.QuantumValue{Value: event})
// 			}
// 			return nil, nil
// 		},
// 		qpool.WithCircuitBreaker("implement", 3, time.Minute),
// 		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
// 		qpool.WithTTL(time.Minute),
// 	)

// 	// Schedule testing after implementation
// 	testResult := pool.Schedule("test",
// 		func() (any, error) {
// 			if err := (<-implementResult).Error; err != nil {
// 				return nil, err
// 			}
// 			for event := range testAgent.Generate(ctx, message) {
// 				broadcast.Send(qpool.QuantumValue{Value: event})
// 			}
// 			return nil, nil
// 		},
// 		qpool.WithCircuitBreaker("test", 3, time.Minute),
// 		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
// 		qpool.WithTTL(time.Minute),
// 	)

// 	// Schedule review after testing
// 	reviewResult := pool.Schedule("review",
// 		func() (any, error) {
// 			if err := (<-testResult).Error; err != nil {
// 				return nil, err
// 			}
// 			for event := range reviewAgent.Generate(ctx, message) {
// 				broadcast.Send(qpool.QuantumValue{Value: event})
// 			}
// 			return nil, nil
// 		},
// 		qpool.WithCircuitBreaker("review", 3, time.Minute),
// 		qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
// 		qpool.WithTTL(time.Minute),
// 	)

// 	// Handle events from the broadcast group
// 	done := make(chan struct{})
// 	go func() {
// 		defer close(done)
// 		for value := range events {
// 			if event, ok := value.Value.(provider.Event); ok {
// 				switch event.Type {
// 				case provider.EventChunk:
// 					if event.Text != "" {
// 						fmt.Print(event.Text)
// 					}
// 				case provider.EventToolCall:
// 					fmt.Printf("\n🛠  Using tool: %s\n", event.Name)
// 				case provider.EventError:
// 					fmt.Printf("\n❌ Error: %s\n", event.Error)
// 				case provider.EventDone:
// 					fmt.Println("\n✅ Development complete")
// 				}
// 			}
// 		}
// 	}()

// 	// Wait for final result or timeout
// 	select {
// 	case result := <-reviewResult:
// 		if result.Error != nil {
// 			fmt.Printf("\n❌ Development failed: %v\n", result.Error)
// 		} else {
// 			fmt.Println("\n=== Development Complete ===")
// 		}
// 	case <-ctx.Done():
// 		fmt.Printf("\n⚠️  Development timed out: %v\n", ctx.Err())
// 	}
// }
