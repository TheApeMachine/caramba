package core

import (
	"context"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/process"
)

// MockLLM implements LLMProvider for testing
type MockLLM struct{}

// MockTool mocks a tool
type MockTool struct{}

// MockMemory mocks a memory
type MockMemory struct{}

// MockProcess mocks a structured output process
type MockProcess struct{}

func (m *MockLLM) GenerateResponse(ctx context.Context, params LLMParams) LLMResponse {
	return LLMResponse{Content: "mock response"}
}

func (m *MockLLM) StreamResponse(ctx context.Context, params LLMParams) <-chan LLMResponse {
	ch := make(chan LLMResponse, 1)
	ch <- LLMResponse{Content: "mock response"}
	close(ch)
	return ch
}

func (m *MockLLM) Name() string {
	return "MockLLM"
}

func (m *MockMemory) QueryAgent() Agent {
	return nil
}

func (m *MockMemory) MutateAgent() Agent {
	return nil
}

func (m *MockMemory) Query(ctx context.Context, proc *process.MemoryLookup) (string, error) {
	return "mock value", nil
}

func (m *MockMemory) Mutate(ctx context.Context, proc *process.MemoryMutate) error {
	return nil
}

func (m *MockMemory) Clear(ctx context.Context) error {
	return nil
}

func (m *MockTool) Name() string {
	return "MockTool"
}

func (m *MockTool) Description() string {
	return "Mock tool for testing"
}

func (m *MockTool) Execute(ctx context.Context, args map[string]any) (any, error) {
	return "mock result", nil
}

func (m *MockTool) Schema() map[string]any {
	return map[string]any{}
}

func (m *MockProcess) Name() string {
	return "MockProcess"
}

func (m *MockProcess) Description() string {
	return "Mock process for testing"
}

func (m *MockProcess) Schema() any {
	return map[string]any{}
}

// Adding the String method to satisfy the process.StructuredOutput interface
func (m *MockProcess) String() string {
	return "MockProcess"
}

// MockMessenger implements Messenger for testing
type MockMessenger struct{}

func (m *MockMessenger) SendDirect(ctx context.Context, to string, content string, messageType string, metadata map[string]interface{}) (string, error) {
	return "message-id", nil
}

func (m *MockMessenger) Publish(ctx context.Context, topic string, content string, messageType string, metadata map[string]interface{}) (string, error) {
	return "message-id", nil
}

func (m *MockMessenger) Broadcast(ctx context.Context, content string, messageType string, metadata map[string]interface{}) ([]string, error) {
	return []string{"message-id"}, nil
}

func (m *MockMessenger) Subscribe(ctx context.Context, topic string) error {
	return nil
}

func (m *MockMessenger) Unsubscribe(ctx context.Context, topic string) error {
	return nil
}

func (m *MockMessenger) CreateTopic(ctx context.Context, name string, description string) error {
	return nil
}

// Fix the GetMessages method to use time.Time instead of interface{}
func (m *MockMessenger) GetMessages(ctx context.Context, since time.Time) ([]Message, error) {
	return []Message{}, nil
}

func (m *MockMessenger) GetTopics(ctx context.Context) ([]Topic, error) {
	return []Topic{}, nil
}

func (m *MockMessenger) GetAgentID() string {
	return "mock-agent"
}

func TestNewAgentBuilder(t *testing.T) {
	Convey("Given a need to create an agent", t, func() {
		Convey("When creating a new agent builder", func() {
			builder := NewAgentBuilder("TestAgent")

			Convey("Then it should not be nil", func() {
				So(builder, ShouldNotBeNil)
			})

			Convey("Then it should have the correct agent name", func() {
				So(builder.agent.name, ShouldEqual, "TestAgent")
			})
		})
	})
}

func TestWithLLM(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the LLM provider", func() {
			llm := &MockLLM{}
			result := builder.WithLLM(llm)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the LLM set", func() {
				So(builder.agent.llm, ShouldEqual, llm)
			})
		})
	})
}

func TestWithSystemPrompt(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the system prompt", func() {
			prompt := "Test system prompt"
			result := builder.WithSystemPrompt(prompt)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the system prompt in messages", func() {
				hasPrompt := false
				for _, msg := range builder.agent.params.Messages {
					if msg.Role == "system" && msg.Content == prompt {
						hasPrompt = true
						break
					}
				}
				So(hasPrompt, ShouldBeTrue)
			})
		})
	})
}

func TestWithIterationLimit(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the iteration limit", func() {
			limit := 5
			result := builder.WithIterationLimit(limit)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the correct iteration limit", func() {
				So(builder.agent.iterationLimit, ShouldEqual, limit)
			})
		})
	})
}

func TestWithMemory(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the memory system", func() {
			memory := &MockMemory{}
			result := builder.WithMemory(memory)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the memory set", func() {
				So(builder.agent.memory, ShouldEqual, memory)
			})
		})
	})
}

func TestWithPlanner(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the planner", func() {
			planner := NewBaseAgent("PlannerAgent")
			result := builder.WithPlanner(planner)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the planner set", func() {
				So(builder.agent.Planner, ShouldEqual, planner)
			})
		})
	})
}

func TestWithOptimizer(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the optimizer", func() {
			optimizer := NewBaseAgent("OptimizerAgent")
			result := builder.WithOptimizer(optimizer)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the optimizer set", func() {
				So(builder.agent.Optimizer, ShouldEqual, optimizer)
			})
		})
	})
}

func TestWithTool(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When adding a tool", func() {
			tool := &MockTool{}
			result := builder.WithTool(tool)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the tool added", func() {
				found := false
				for _, t := range builder.agent.params.Tools {
					if t == tool {
						found = true
						break
					}
				}
				So(found, ShouldBeTrue)
			})
		})
	})
}

func TestWithProcess(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the process", func() {
			proc := &MockProcess{}
			result := builder.WithProcess(proc)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the process name set", func() {
				So(builder.agent.params.ResponseFormatName, ShouldEqual, proc.Name())
			})

			Convey("Then the agent should have the process description set", func() {
				So(builder.agent.params.ResponseFormatDescription, ShouldEqual, proc.Description())
			})

			Convey("Then the agent should have the process schema set", func() {
				So(builder.agent.params.Schema, ShouldEqual, proc.Schema())
			})
		})
	})
}

func TestWithMessenger(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When setting the messenger", func() {
			messenger := &MockMessenger{}
			result := builder.WithMessenger(messenger)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have the messenger set", func() {
				So(builder.agent.Messenger, ShouldEqual, messenger)
			})
		})
	})
}

func TestWithStreaming(t *testing.T) {
	Convey("Given an agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When enabling streaming", func() {
			result := builder.WithStreaming(true)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have streaming enabled", func() {
				So(builder.agent.streaming, ShouldBeTrue)
			})
		})

		Convey("When disabling streaming", func() {
			result := builder.WithStreaming(false)

			Convey("Then it should return the builder for chaining", func() {
				So(result, ShouldEqual, builder)
			})

			Convey("Then the agent should have streaming disabled", func() {
				So(builder.agent.streaming, ShouldBeFalse)
			})
		})
	})
}

func TestBuild(t *testing.T) {
	Convey("Given a configured agent builder", t, func() {
		builder := NewAgentBuilder("TestAgent")

		Convey("When building the agent", func() {
			agent := builder.Build()

			Convey("Then it should return a non-nil agent", func() {
				So(agent, ShouldNotBeNil)
			})

			Convey("Then the agent should have the correct name", func() {
				So(agent.Name(), ShouldEqual, "TestAgent")
			})
		})
	})
}
