package tests

import (
	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/types"
)

type MockGenerator struct {
	ctx         types.Context
	accumulator *stream.Accumulator
	status      types.AgentStatus
	agents      map[string]types.Generator
}

func NewMockGenerator() *MockGenerator {
	return &MockGenerator{
		ctx: agent.NewContext(agent.NewConfig(
			"test", "test", "test", tools.NewToolset().String(),
		)),
		accumulator: stream.NewAccumulator(),
		status:      types.AgentStatusIdle,
		agents:      make(map[string]types.Generator),
	}
}

func (mock *MockGenerator) Generate(
	message *provider.Message,
) <-chan *provider.Event {
	out := make(chan *provider.Event)

	chunks := []*provider.Event{
		provider.NewEvent(
			"test",
			provider.EventChunk,
			"Hello, ",
			"",
			map[string]any{},
		),
		provider.NewEvent(
			"test",
			provider.EventChunk,
			"world!",
			"",
			map[string]any{},
		),
		provider.NewEvent(
			"test",
			provider.EventChunk,
			"```json\n{\n  \"tool\": \"break\",\n  \"args\": {\n	\"final_answer\": \"test\"\n}\n}\n```\n",
			"",
			map[string]any{},
		),
		provider.NewEvent(
			"test",
			provider.EventStop,
			"",
			"",
			map[string]any{},
		),
	}

	go func() {
		defer close(out)

		for _, chunk := range chunks {
			out <- chunk
			mock.accumulator.Append(chunk.Text)
		}
	}()

	return out
}

func (mock *MockGenerator) Status() types.AgentStatus {
	return mock.status
}

func (mock *MockGenerator) SetStatus(status types.AgentStatus) {
	mock.status = status
}

func (mock *MockGenerator) Accumulator() *stream.Accumulator {
	return mock.accumulator
}

func (mock *MockGenerator) Ctx() types.Context {
	return mock.ctx
}

func (mock *MockGenerator) Agents() map[string]types.Generator {
	return mock.agents
}
