package core

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewMessage(t *testing.T) {
	Convey("Given parameters for a new message", t, func() {
		sender := "agent1"
		recipient := "agent2"
		topic := ""
		content := "Hello"
		messageType := MessageTypeText
		metadata := map[string]interface{}{"key": "value"}

		Convey("When creating a new message", func() {
			message := NewMessage(sender, recipient, topic, content, messageType, metadata)

			Convey("Then it should have the correct properties", func() {
				So(message.ID, ShouldNotBeEmpty)
				So(message.Sender, ShouldEqual, sender)
				So(message.Recipient, ShouldEqual, recipient)
				So(message.Topic, ShouldEqual, topic)
				So(message.Content, ShouldEqual, content)
				So(message.Type, ShouldEqual, messageType)
				So(message.Metadata["key"], ShouldEqual, "value")
			})
		})

		Convey("When creating a message with empty message type", func() {
			message := NewMessage(sender, recipient, topic, content, "", metadata)

			Convey("Then it should default to text message type", func() {
				So(message.Type, ShouldEqual, MessageTypeText)
			})
		})

		Convey("When creating a message with nil metadata", func() {
			message := NewMessage(sender, recipient, topic, content, messageType, nil)

			Convey("Then it should initialize an empty metadata map", func() {
				So(message.Metadata, ShouldNotBeNil)
				So(len(message.Metadata), ShouldEqual, 0)
			})
		})
	})
}

func TestMessageTypes(t *testing.T) {
	Convey("Given different types of messages", t, func() {
		Convey("When creating a direct message", func() {
			message := NewMessage("agent1", "agent2", "", "Hello", MessageTypeText, nil)

			Convey("Then it should be identified as a direct message", func() {
				So(message.IsDirect(), ShouldBeTrue)
				So(message.IsTopic(), ShouldBeFalse)
				So(message.IsBroadcast(), ShouldBeFalse)
			})
		})

		Convey("When creating a topic message", func() {
			message := NewMessage("agent1", "", "topic1", "Hello", MessageTypeText, nil)

			Convey("Then it should be identified as a topic message", func() {
				So(message.IsDirect(), ShouldBeFalse)
				So(message.IsTopic(), ShouldBeTrue)
				So(message.IsBroadcast(), ShouldBeFalse)
			})
		})

		Convey("When creating a broadcast message", func() {
			message := NewMessage("agent1", "", "", "Hello", MessageTypeText, nil)

			Convey("Then it should be identified as a broadcast message", func() {
				So(message.IsDirect(), ShouldBeFalse)
				So(message.IsTopic(), ShouldBeFalse)
				So(message.IsBroadcast(), ShouldBeTrue)
			})
		})
	})
}

func TestNewTopic(t *testing.T) {
	Convey("Given parameters for a new topic", t, func() {
		name := "test-topic"
		description := "A test topic"
		creator := "agent1"

		Convey("When creating a new topic", func() {
			topic := NewTopic(name, description, creator)

			Convey("Then it should have the correct properties", func() {
				So(topic.Name, ShouldEqual, name)
				So(topic.Description, ShouldEqual, description)
				So(topic.Creator, ShouldEqual, creator)
				So(len(topic.Subscribers), ShouldEqual, 1)
				So(topic.Subscribers[0], ShouldEqual, creator)
			})
		})
	})
}

func TestTopicSubscribers(t *testing.T) {
	Convey("Given a topic with a creator", t, func() {
		topic := NewTopic("test-topic", "A test topic", "agent1")

		Convey("When checking if the creator is a subscriber", func() {
			hasSubscriber := topic.HasSubscriber("agent1")

			Convey("Then it should return true", func() {
				So(hasSubscriber, ShouldBeTrue)
			})
		})

		Convey("When checking if a non-subscriber is a subscriber", func() {
			hasSubscriber := topic.HasSubscriber("agent2")

			Convey("Then it should return false", func() {
				So(hasSubscriber, ShouldBeFalse)
			})
		})

		Convey("When adding a new subscriber", func() {
			added := topic.AddSubscriber("agent2")

			Convey("Then it should return true and add the subscriber", func() {
				So(added, ShouldBeTrue)
				So(len(topic.Subscribers), ShouldEqual, 2)
				So(topic.HasSubscriber("agent2"), ShouldBeTrue)
			})
		})

		Convey("When adding an existing subscriber", func() {
			added := topic.AddSubscriber("agent1")

			Convey("Then it should return false and not duplicate the subscriber", func() {
				So(added, ShouldBeFalse)
				So(len(topic.Subscribers), ShouldEqual, 1)
			})
		})

		Convey("When removing an existing subscriber", func() {
			removed := topic.RemoveSubscriber("agent1")

			Convey("Then it should return true and remove the subscriber", func() {
				So(removed, ShouldBeTrue)
				So(len(topic.Subscribers), ShouldEqual, 0)
				So(topic.HasSubscriber("agent1"), ShouldBeFalse)
			})
		})

		Convey("When removing a non-existent subscriber", func() {
			removed := topic.RemoveSubscriber("agent2")

			Convey("Then it should return false", func() {
				So(removed, ShouldBeFalse)
				So(len(topic.Subscribers), ShouldEqual, 1)
			})
		})
	})
}

func TestNewMessageRegistry(t *testing.T) {
	Convey("Given a need for a message registry", t, func() {
		Convey("When creating a new message registry", func() {
			registry := NewMessageRegistry()

			Convey("Then it should not be nil", func() {
				So(registry, ShouldNotBeNil)
			})

			Convey("Then it should have empty maps and slices", func() {
				So(registry.messengers, ShouldNotBeNil)
				So(len(registry.messengers), ShouldEqual, 0)
				So(registry.topics, ShouldNotBeNil)
				So(len(registry.topics), ShouldEqual, 0)
				So(registry.messages, ShouldNotBeNil)
				So(len(registry.messages), ShouldEqual, 0)
			})
		})
	})
}

func TestNewInMemoryMessenger(t *testing.T) {
	Convey("Given a need for an in-memory messenger", t, func() {
		// Save the original registry to restore later
		originalRegistry := globalRegistry
		defer func() { globalRegistry = originalRegistry }()

		// Create a new registry for this test
		globalRegistry = NewMessageRegistry()

		Convey("When creating a new in-memory messenger", func() {
			messenger := NewInMemoryMessenger("agent1")

			Convey("Then it should not be nil", func() {
				So(messenger, ShouldNotBeNil)
			})

			Convey("Then it should have the correct agent ID", func() {
				So(messenger.agentID, ShouldEqual, "agent1")
			})

			Convey("Then it should be registered in the global registry", func() {
				So(globalRegistry.messengers["agent1"], ShouldEqual, messenger)
			})
		})
	})
}

func TestGenerateMessageID(t *testing.T) {
	Convey("Given a need for a message ID", t, func() {
		Convey("When generating a message ID", func() {
			id1 := generateMessageID()
			id2 := generateMessageID()

			Convey("Then it should return a non-empty string", func() {
				So(id1, ShouldNotBeEmpty)
			})

			Convey("Then consecutive IDs should be different", func() {
				So(id1, ShouldNotEqual, id2)
			})
		})
	})
}
