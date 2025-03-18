package ai

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/event"
	"github.com/theapemachine/caramba/pkg/message"
)

// testEvent creates a test event artifact with predefined data
func testEvent() *event.Artifact {
	msg, err := message.New(
		message.UserRole,
		"test-name",
		"test-content",
	).Message().Marshal()

	if errnie.Error(err) != nil {
		errnie.Error("failed to create message artifact", "error", err)
		return nil
	}

	return event.New(
		"test",
		event.MessageEvent,
		event.UserRole,
		msg,
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
	messages := agent.params.Messages
	So(messages, ShouldNotBeNil)
	So(len(messages), ShouldEqual, len(expected))

	for i, exp := range expected {
		msg := messages[i]

		content := msg.Content
		So(content, ShouldEqual, exp.content)

		role := msg.Role
		So(role, ShouldEqual, exp.role)

		name := msg.Name
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
			msg := message.New(
				message.UserRole,
				"test-name",
				"test-content",
			)

			msg2, err := msg.Message().Marshal()
			So(err, ShouldBeNil)

			evt := event.New(
				"test",
				event.MessageEvent,
				event.UserRole,
				msg2,
			)
			writeEventToAgent(agent, evt)

			verifyMessages(agent, []messageVerification{
				{content: "test-content", role: "user", name: "test-name"},
			})
		})

		Convey("When writing multiple message events", func() {
			// First message
			msg := message.New(
				message.UserRole,
				"test-name",
				"test-content",
			)

			msg1, err := msg.Message().Marshal()
			So(err, ShouldBeNil)

			evt1 := event.New(
				"test",
				event.MessageEvent,
				event.UserRole,
				msg1,
			)
			writeEventToAgent(agent, evt1)

			// Second message
			msg2 := message.New(
				message.AssistantRole,
				"test-name-2",
				"test-content-2",
			)

			msg22, err := msg2.Message().Marshal()
			So(err, ShouldBeNil)

			evt2 := event.New(
				"test",
				event.MessageEvent,
				event.AssistantRole,
				msg22,
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
