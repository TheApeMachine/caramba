package ai

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
)

// testEvent creates a test event artifact with predefined data
func testEvent() *event.Artifact {
	return event.New(
		"test",
		event.MessageEvent,
		event.UserRole,
		[]byte("test-data"),
	)
}

// testMessage creates a message artifact for testing
func testMessage() *message.Artifact {
	return message.New(
		message.UserRole,
		"test-name",
		"test-content",
	)
}

// writeEventToAgent is a helper to write an event to an agent and verify the write
func writeEventToAgent(agent *Agent, evt *event.Artifact) {
	nn, err := io.Copy(agent, evt)
	So(err, ShouldBeNil)
	So(nn, ShouldBeGreaterThan, 0)
}

type messageVerification struct {
	content string
	role    string
	name    string
}

// verifyMessages checks that the agent's context contains the expected messages
func verifyMessages(agent *Agent, expected []messageVerification) {
	messages, err := agent.params.Messages()
	So(err, ShouldBeNil)
	So(messages.Len(), ShouldEqual, len(expected))

	for i, exp := range expected {
		msg := messages.At(i)

		content, err := msg.Content()
		So(err, ShouldBeNil)
		So(content, ShouldEqual, exp.content)

		role, err := msg.Role()
		So(err, ShouldBeNil)
		So(role, ShouldEqual, exp.role)

		name, err := msg.Name()
		So(err, ShouldBeNil)
		So(name, ShouldEqual, exp.name)
	}
}

func TestNewAgent(t *testing.T) {
	Convey("Given a new agent", t, func() {
		agent := NewAgent()

		So(agent, ShouldNotBeNil)
		So(agent.params, ShouldNotBeNil)
		So(agent.buffer, ShouldNotBeNil)
	})
}

func TestRead(t *testing.T) {
	Convey("Given a new agent", t, func() {
		agent := NewAgent()

		Convey("When reading from the agent with data", func() {
			// Write test event first
			writeEventToAgent(agent, testEvent())

			// Read from the agent
			p := make([]byte, 1024)
			n, err := agent.Read(p)
			So(err, ShouldEqual, io.EOF)
			So(n, ShouldBeGreaterThan, 0)
		})

		Convey("When reading without data", func() {
			p := make([]byte, 1024)
			n, err := agent.Read(p)
			So(err, ShouldEqual, io.EOF)
			So(n, ShouldBeGreaterThan, 0) // Context always has some data due to parameters
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given a new agent", t, func() {
		agent := NewAgent()

		Convey("When writing valid event data", func() {
			writeEventToAgent(agent, testEvent())
		})

		Convey("When writing empty data", func() {
			n, err := agent.Write([]byte{})
			So(err, ShouldBeError)
			So(err.Error(), ShouldEqual, "empty input")
			So(n, ShouldEqual, 0)
		})

		Convey("When writing a message event", func() {
			evt := event.New(
				"test",
				event.MessageEvent,
				event.UserRole,
				testMessage().Marshal(),
			)
			writeEventToAgent(agent, evt)

			verifyMessages(agent, []messageVerification{
				{content: "test-content", role: "user", name: "test-name"},
			})
		})

		Convey("When writing multiple message events", func() {
			// First message
			evt1 := event.New(
				"test",
				event.MessageEvent,
				event.UserRole,
				testMessage().Marshal(),
			)
			writeEventToAgent(agent, evt1)

			// Second message
			msg2 := message.New(
				message.AssistantRole,
				"test-name-2",
				"test-content-2",
			)
			evt2 := event.New(
				"test",
				event.MessageEvent,
				event.AssistantRole,
				msg2.Marshal(),
			)
			writeEventToAgent(agent, evt2)

			verifyMessages(agent, []messageVerification{
				{content: "test-content", role: "user", name: "test-name"},
				{content: "test-content-2", role: "assistant", name: "test-name-2"},
			})
		})
	})
}

func TestClose(t *testing.T) {
	Convey("Given a new agent", t, func() {
		agent := NewAgent()

		Convey("When closing the agent", func() {
			err := agent.Close()
			So(err, ShouldBeNil)
		})
	})
}
