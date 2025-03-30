package ai

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/provider"
)

func init() {
	// Set up Viper configuration for testing
	viper.Set("settings.defaults.provider", "openai")
	viper.Set("models.openai", "gpt-4")
	viper.Set("settings.defaults.temperature", 0.7)
	viper.Set("settings.defaults.stream", true)
}

// testEvent creates a test event artifact with predefined data
func testEvent() *datura.Artifact {
	params := provider.NewParams()
	params.Messages = append(params.Messages, &provider.Message{
		Content: "test-content",
		Role:    provider.MessageRoleUser,
		Name:    "test-name",
	})
	return datura.New(
		datura.WithPayload(params.Marshal()),
	)
}

// writeEventToAgent is a helper to write an event to an agent and verify the write
func writeEventToAgent(agent *Agent, evt *datura.Artifact) {
	nn, err := io.Copy(agent, evt)
	So(err, ShouldBeNil)
	So(nn, ShouldBeGreaterThan, 0)
}

type messageVerification struct {
	content string
	role    provider.MessageRole
	name    string
}

// verifyMessages checks that the agent's context contains the expected messages
func verifyMessages(agent *Agent, expected []messageVerification) {
	messages := agent.params.Messages
	So(messages, ShouldNotBeNil)
	So(len(messages), ShouldEqual, len(expected))

	for i, exp := range expected {
		msg := messages[i]
		So(msg, ShouldNotBeNil)
		So(msg.Content, ShouldEqual, exp.content)
		So(msg.Role, ShouldEqual, exp.role)
		So(msg.Name, ShouldEqual, exp.name)
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
			verifyMessages(agent, []messageVerification{
				{content: "test-content", role: provider.MessageRoleUser, name: "test-name"},
			})
		})

		Convey("When writing empty data", func() {
			n, err := agent.Write([]byte{})
			So(err, ShouldBeError)
			So(err.Error(), ShouldEqual, "empty input")
			So(n, ShouldEqual, 0)
		})

		Convey("When writing a message event", func() {
			params := provider.NewParams()
			params.Messages = append(params.Messages, &provider.Message{
				Content: "test-content",
				Role:    provider.MessageRoleUser,
				Name:    "test-name",
			})
			evt := datura.New(
				datura.WithPayload(params.Marshal()),
			)
			writeEventToAgent(agent, evt)

			verifyMessages(agent, []messageVerification{
				{content: "test-content", role: provider.MessageRoleUser, name: "test-name"},
			})
		})

		Convey("When writing multiple message events", func() {
			params1 := provider.NewParams()
			params1.Messages = append(params1.Messages, &provider.Message{
				Content: "test-content",
				Role:    provider.MessageRoleUser,
				Name:    "test-name",
			})
			evt1 := datura.New(
				datura.WithPayload(params1.Marshal()),
			)
			writeEventToAgent(agent, evt1)

			params2 := provider.NewParams()
			params2.Messages = append(params2.Messages, &provider.Message{
				Content: "test-content-2",
				Role:    provider.MessageRoleAssistant,
				Name:    "test-name-2",
			})
			evt2 := datura.New(
				datura.WithPayload(params2.Marshal()),
			)
			writeEventToAgent(agent, evt2)

			verifyMessages(agent, []messageVerification{
				{content: "test-content", role: provider.MessageRoleUser, name: "test-name"},
				{content: "test-content-2", role: provider.MessageRoleAssistant, name: "test-name-2"},
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
