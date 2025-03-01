package core

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/hub"
)

// MockMultiResponseLLM mocks an LLM provider that tracks number of calls
type MockMultiResponseLLM struct {
	MockLLM
	callCount int
}

func (m *MockMultiResponseLLM) GenerateResponse(ctx context.Context, params LLMParams) LLMResponse {
	m.callCount++
	return LLMResponse{Content: "Iteration " + string(rune('0'+m.callCount))}
}

func TestNewBaseAgent(t *testing.T) {
	Convey("Given a need for a base agent", t, func() {
		Convey("When creating a new base agent", func() {
			agent := NewBaseAgent("TestAgent")

			Convey("Then it should not be nil", func() {
				So(agent, ShouldNotBeNil)
			})

			Convey("Then it should have the correct name", func() {
				So(agent.name, ShouldEqual, "TestAgent")
			})

			Convey("Then it should have default parameters", func() {
				So(agent.params, ShouldNotBeNil)
				So(agent.params.Model, ShouldEqual, "gpt-4o-mini")
				So(agent.params.Temperature, ShouldEqual, 0.7)
				So(agent.params.MaxTokens, ShouldEqual, 1024)
				So(len(agent.params.Tools), ShouldEqual, 0)
				So(len(agent.params.Messages), ShouldEqual, 0)
			})

			Convey("Then it should have default status", func() {
				So(agent.status, ShouldEqual, AgentStatusIdle)
			})

			Convey("Then it should have default iteration limit", func() {
				So(agent.iterationLimit, ShouldEqual, 1)
			})

			Convey("Then it should have streaming disabled by default", func() {
				So(agent.streaming, ShouldBeFalse)
			})

			Convey("Then it should have a messenger", func() {
				So(agent.Messenger, ShouldNotBeNil)
			})
		})
	})
}

func TestBaseAgentGetters(t *testing.T) {
	Convey("Given a base agent", t, func() {
		agent := NewBaseAgent("TestAgent")

		Convey("When getting the name", func() {
			name := agent.Name()

			Convey("Then it should return the correct name", func() {
				So(name, ShouldEqual, "TestAgent")
			})
		})

		Convey("When getting the memory", func() {
			memory := agent.Memory()

			Convey("Then it should return nil by default", func() {
				So(memory, ShouldBeNil)
			})
		})

		Convey("When getting the iteration limit", func() {
			limit := agent.IterationLimit()

			Convey("Then it should return the default limit", func() {
				So(limit, ShouldEqual, 1)
			})
		})

		Convey("When getting the LLM", func() {
			llm := agent.LLM()

			Convey("Then it should return nil by default", func() {
				So(llm, ShouldBeNil)
			})
		})

		Convey("When getting the streaming flag", func() {
			streaming := agent.Streaming()

			Convey("Then it should return false by default", func() {
				So(streaming, ShouldBeFalse)
			})
		})

		Convey("When getting the parameters", func() {
			params := agent.Params()

			Convey("Then it should return the parameters", func() {
				So(params, ShouldNotBeNil)
				So(params, ShouldEqual, agent.params)
			})
		})

		Convey("When getting the status", func() {
			status := agent.Status()

			Convey("Then it should return the default status", func() {
				So(status, ShouldEqual, AgentStatusIdle)
			})
		})

		Convey("When getting the messenger", func() {
			messenger := agent.GetMessenger()

			Convey("Then it should return the messenger", func() {
				So(messenger, ShouldNotBeNil)
				So(messenger, ShouldEqual, agent.Messenger)
			})
		})
	})
}

func TestBaseAgentSetters(t *testing.T) {
	Convey("Given a base agent", t, func() {
		agent := NewBaseAgent("TestAgent")

		Convey("When setting the status", func() {
			agent.SetStatus(AgentStatusRunning)

			Convey("Then the status should be updated", func() {
				So(agent.status, ShouldEqual, AgentStatusRunning)
			})
		})

		Convey("When adding a tool", func() {
			tool := &MockTool{}
			agent.AddTool(tool)

			Convey("Then the tool should be added to the parameters", func() {
				So(len(agent.params.Tools), ShouldEqual, 1)
				So(agent.params.Tools[0], ShouldEqual, tool)
			})
		})

		Convey("When adding a user message", func() {
			agent.AddUserMessage("Hello")

			Convey("Then the message should be added to the parameters", func() {
				So(len(agent.params.Messages), ShouldEqual, 1)
				So(agent.params.Messages[0].Role, ShouldEqual, "user")
				So(agent.params.Messages[0].Content, ShouldEqual, "Hello")
			})
		})

		Convey("When adding an assistant message", func() {
			agent.AddAssistantMessage("Hi there")

			Convey("Then the message should be added to the parameters", func() {
				So(len(agent.params.Messages), ShouldEqual, 1)
				So(agent.params.Messages[0].Role, ShouldEqual, "assistant")
				So(agent.params.Messages[0].Content, ShouldEqual, "Hi there")
			})
		})

		Convey("When setting the memory", func() {
			memory := &MockMemory{}
			agent.SetMemory(memory)

			Convey("Then the memory should be updated", func() {
				So(agent.memory, ShouldEqual, memory)
			})
		})

		Convey("When setting the planner", func() {
			planner := NewBaseAgent("PlannerAgent")
			agent.SetPlanner(planner)

			Convey("Then the planner should be updated", func() {
				So(agent.Planner, ShouldEqual, planner)
			})
		})

		Convey("When setting the optimizer", func() {
			optimizer := NewBaseAgent("OptimizerAgent")
			agent.SetOptimizer(optimizer)

			Convey("Then the optimizer should be updated", func() {
				So(agent.Optimizer, ShouldEqual, optimizer)
			})
		})

		Convey("When setting the LLM", func() {
			llm := &MockLLM{}
			agent.SetLLM(llm)

			Convey("Then the LLM should be updated", func() {
				So(agent.llm, ShouldEqual, llm)
			})
		})

		Convey("When setting the system prompt", func() {
			agent.SetSystemPrompt("You are a helpful assistant")

			Convey("Then a system message should be added to the parameters", func() {
				So(len(agent.params.Messages), ShouldEqual, 1)
				So(agent.params.Messages[0].Role, ShouldEqual, "system")
				So(agent.params.Messages[0].Content, ShouldEqual, "You are a helpful assistant")
			})
		})

		Convey("When setting the process", func() {
			proc := &MockProcess{}
			agent.SetProcess(proc)

			Convey("Then the process parameters should be updated", func() {
				So(agent.params.ResponseFormatName, ShouldEqual, proc.Name())
				So(agent.params.ResponseFormatDescription, ShouldEqual, proc.Description())
				So(agent.params.Schema, ShouldEqual, proc.Schema())
			})
		})

		Convey("When setting the model", func() {
			agent.SetModel("gpt-4")

			Convey("Then the model should be updated", func() {
				So(agent.params.Model, ShouldEqual, "gpt-4")
			})
		})

		Convey("When setting the temperature", func() {
			agent.SetTemperature(0.5)

			Convey("Then the temperature should be updated", func() {
				So(agent.params.Temperature, ShouldEqual, 0.5)
			})
		})

		Convey("When setting the iteration limit", func() {
			agent.SetIterationLimit(3)

			Convey("Then the iteration limit should be updated", func() {
				So(agent.iterationLimit, ShouldEqual, 3)
			})
		})

		Convey("When setting the messenger", func() {
			messenger := NewInMemoryMessenger("TestAgent2")
			agent.SetMessenger(messenger)

			Convey("Then the messenger should be updated", func() {
				So(agent.Messenger, ShouldEqual, messenger)
			})
		})

		Convey("When setting the streaming flag", func() {
			agent.SetStreaming(true)

			Convey("Then the streaming flag should be updated", func() {
				So(agent.streaming, ShouldBeTrue)
			})
		})
	})
}

func TestGetTool(t *testing.T) {
	Convey("Given a base agent with tools", t, func() {
		agent := NewBaseAgent("TestAgent")
		tool1 := &MockTool{}
		tool2 := &MockToolWithName{name: "special-tool"}
		agent.AddTool(tool1)
		agent.AddTool(tool2)

		Convey("When getting a tool by name that exists", func() {
			foundTool := agent.GetTool("special-tool")

			Convey("Then it should return the correct tool", func() {
				So(foundTool, ShouldEqual, tool2)
			})
		})

		Convey("When getting a tool by name that doesn't exist", func() {
			foundTool := agent.GetTool("non-existent-tool")

			Convey("Then it should return nil", func() {
				So(foundTool, ShouldBeNil)
			})
		})
	})
}

// MockToolWithName is a mock tool with a custom name
type MockToolWithName struct {
	MockTool
	name string
}

func (m *MockToolWithName) Name() string {
	return m.name
}

// TestBaseAgentIterations tests that the BaseAgent correctly handles multiple iterations
func TestBaseAgentIterations(t *testing.T) {
	Convey("Given a base agent with a higher iteration limit", t, func() {
		agent := NewBaseAgent("TestAgent")
		agent.SetIterationLimit(3)

		// Create a mock LLM that counts calls
		mockLLM := &MockMultiResponseLLM{}
		agent.SetLLM(mockLLM)

		Convey("When handling an event that triggers iterations", func() {
			event := hub.NewEvent(
				"user",
				"user",
				"message",
				hub.EventTypeMessage,
				"Run multiple iterations",
				map[string]string{},
			)

			err := agent.handleEvent(context.Background(), event)

			Convey("Then it should not error", func() {
				So(err, ShouldBeNil)
			})

			Convey("Then it should run the specified number of iterations", func() {
				So(mockLLM.callCount, ShouldEqual, 3)
			})
		})
	})
}
