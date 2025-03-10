package ai

import (
	"encoding/json"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// TestNewAgent tests the NewAgent constructor
func TestNewAgent(t *testing.T) {
	Convey("Given parameters for a new Agent", t, func() {
		Convey("When creating a new Agent", func() {
			agent := NewAgent()

			Convey("Then the agent should have the correct properties", func() {
				So(agent, ShouldNotBeNil)
				So(agent.Context, ShouldNotBeNil)
				So(agent.in, ShouldNotBeNil)
				So(agent.out, ShouldNotBeNil)
				So(agent.enc, ShouldNotBeNil)
				So(agent.dec, ShouldNotBeNil)
			})
		})
	})
}

// TestAgentRead tests the Read method of Agent
func TestAgentRead(t *testing.T) {
	Convey("Given an Agent with encoded data", t, func() {
		agent := NewAgent()
		buffer := make([]byte, 1024)

		Convey("When reading from the agent", func() {
			n, err := agent.Read(buffer)

			Convey("Then it should return valid JSON data", func() {
				So(err, ShouldBeNil)
				So(n, ShouldBeGreaterThan, 0)

				// Verify it contains ContextData structure
				jsonStr := string(buffer[:n])
				So(jsonStr, ShouldContainSubstring, "messages")
				So(jsonStr, ShouldContainSubstring, "model")
			})
		})

		Convey("When reading after buffer is depleted", func() {
			// First read to consume buffer
			firstBuffer := make([]byte, 1024)
			agent.Read(firstBuffer)

			// Second read should indicate buffer is empty
			n, err := agent.Read(buffer)

			Convey("Then it should return EOF", func() {
				So(err, ShouldEqual, io.EOF)
				So(n, ShouldEqual, 0)
			})
		})
	})
}

// TestAgentWrite tests the Write method of Agent
func TestAgentWrite(t *testing.T) {
	Convey("Given an Agent", t, func() {
		agent := NewAgent()

		Convey("When writing a valid event with a message", func() {
			message := core.NewMessage("user", "testuser", "test content")
			event := core.NewEvent(message, nil)

			// Serialize the event to JSON
			jsonData, err := json.Marshal(event)
			So(err, ShouldBeNil)

			n, err := agent.Write(jsonData)

			Convey("Then it should update the agent's context with the message", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(jsonData))
				So(len(agent.Context.Messages), ShouldEqual, 1)
				So(agent.Context.Messages[0].Role, ShouldEqual, "user")
				So(agent.Context.Messages[0].Name, ShouldEqual, "testuser")
				So(agent.Context.Messages[0].Content, ShouldEqual, "test content")
			})
		})

		Convey("When writing an event without a message", func() {
			event := core.NewEvent(nil, nil)

			// Serialize the event to JSON
			jsonData, err := json.Marshal(event)
			So(err, ShouldBeNil)

			n, err := agent.Write(jsonData)

			Convey("Then it should return a validation error", func() {
				So(err, ShouldNotBeNil)
				So(err, ShouldHaveSameTypeAs, errnie.NewErrValidation(""))
				So(n, ShouldEqual, len(jsonData))
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"broken": "json"`)
			n, err := agent.Write(invalidJSON)

			Convey("Then it should not return an error but not update the context", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
				So(len(agent.Context.Messages), ShouldEqual, 0)
			})
		})
	})
}

// TestAgentClose tests the Close method of Agent
func TestAgentClose(t *testing.T) {
	Convey("Given an Agent", t, func() {
		agent := NewAgent()

		Convey("When closing the agent", func() {
			err := agent.Close()

			Convey("Then it should close the context", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}
