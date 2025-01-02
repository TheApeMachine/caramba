package examples

import (
	"context"
	"time"

	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/qpool"
)

/*
Strawberry is an example of the infamous strawberry problem, where the user
asks the agent to count the number of times the letter "r" appears in the word
"strawberry". This is an especially good example, since all LLMs will get this
wrong, and it's a good way to test the agent's ability to reason.
*/
type Strawberry struct {
	ctx    context.Context
	agents map[string][]*ai.Agent
}

/*
NewStrawberry creates a new Strawberry instance with the specified context and role.
It initializes an empty scratchpad for accumulating assistant responses
and sets up the basic formatting configuration.

Parameters:
  - ctx: The context for operations
  - role: The role designation for the AI agent
*/
func NewStrawberry(ctx context.Context, role string) *Strawberry {
	return &Strawberry{
		ctx: ctx,
		agents: map[string][]*ai.Agent{
			"prompt":     {ai.NewAgent(ctx, "prompt", 1)},
			"reasoner":   {ai.NewAgent(ctx, "reasoner", 2)},
			"challenger": {ai.NewAgent(ctx, "challenger", 2)},
			"solver":     {ai.NewAgent(ctx, "solver", 2)},
		},
	}
}

/*
Run starts the strawberry loop, which allows the user to interact with the agent.
It initializes the agents and creates a worker pool to handle the different
steps of the strawberry problem.
*/
func (strawberry *Strawberry) Run() error {
	// Configure and create the worker pool
	config := &qpool.Config{
		SchedulingTimeout: time.Second * 60,
	}
	pool := qpool.NewQ(strawberry.ctx, 2, 4, config)
	defer pool.Close()

	// Create a broadcast group for events
	broadcast := pool.CreateBroadcastGroup("strawberry-events", time.Minute)
	// events := pool.Subscribe("strawberry-events")

	// Create and configure the conversation thread
	message := provider.NewMessage(provider.RoleUser, `How many times do we find the letter r in the word strawberry?`)
	thread := provider.NewThread()
	thread.AddMessage(message)

	for _, agents := range strawberry.agents {
		for _, agent := range agents {
			pool.Schedule(agent.Identity.Name,
				func() (any, error) {
					accumulator := stream.NewAccumulator()
					for event := range accumulator.Generate(
						strawberry.ctx,
						agent.Generate(strawberry.ctx, message),
					) {
						broadcast.Send(qpool.QuantumValue{Value: event})
					}

					return nil, nil
				},
				qpool.WithCircuitBreaker(agent.Identity.Name, 3, time.Minute),
				qpool.WithRetry(3, &qpool.ExponentialBackoff{Initial: time.Second}),
				qpool.WithTTL(time.Minute),
			)
		}
	}

	return nil

}
