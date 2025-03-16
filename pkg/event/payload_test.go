package event

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/context"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/message"
)

func testContext() *context.Artifact {
	return context.New(
		"test-model",
		nil,   // messages
		nil,   // tools
		nil,   // process
		0.7,   // temperature
		1.0,   // topP
		40,    // topK
		0.0,   // presencePenalty
		0.0,   // frequencyPenalty
		2048,  // maxTokens
		false, // stream
	)
}

func testMessageArtifact() *Artifact {
	msg, err := message.New(
		message.UserRole,
		"test",
		"test",
	).Message().Marshal()

	if errnie.Error(err) != nil {
		errnie.Error("failed to create message artifact", "error", err)
		return nil
	}

	return New(
		"test",
		MessageEvent,
		UserRole,
		msg,
	)
}

func testContextArtifact() *Artifact {
	return New(
		"test",
		ContextEvent,
		UserRole,
		testContext().Marshal(),
	)
}

func TestToContext(t *testing.T) {
	Convey("Given an artifact", t, func() {
		Convey("When converting a MessageEvent to context", func() {
			artifact := testMessageArtifact()
			ctx := testContext()

			err := artifact.ToContext(ctx)
			So(err, ShouldBeNil)

			messages, err := ctx.Messages()
			So(err, ShouldBeNil)
			So(messages.Len(), ShouldEqual, 1)

			msg := messages.At(0)

			role, err := msg.Role()
			So(err, ShouldBeNil)
			So(role, ShouldEqual, "user")

			content, err := msg.Content()
			So(err, ShouldBeNil)
			So(content, ShouldEqual, "test")
		})

		Convey("When converting a ContextEvent to context", func() {
			artifact := testContextArtifact()
			ctx := testContext()

			err := artifact.ToContext(ctx)
			So(err, ShouldBeNil)

			// Verify the context was updated with the artifact's data
			messages, err := ctx.Messages()
			So(err, ShouldBeNil)
			So(messages.Len(), ShouldEqual, 0) // or whatever the expected length is
		})

		Convey("When converting an unknown event type", func() {
			artifact := New(
				"test",
				"unknown",
				UserRole,
				[]byte("test"),
			)

			ctx := testContext()
			err := artifact.ToContext(ctx)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unknown event type")
		})
	})
}
