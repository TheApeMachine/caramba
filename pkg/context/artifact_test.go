package context

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func testArtifact() *Artifact {
	return New(
		"test-model",
		nil, // messages
		nil, // tools
		[]byte("test-process"),
		0.7,   // temperature
		1.0,   // topP
		40,    // topK
		0.0,   // presencePenalty
		0.0,   // frequencyPenalty
		2048,  // maxTokens
		false, // stream
	)
}

func TestNewArtifact(t *testing.T) {
	Convey("Given a new artifact", t, func() {
		artifact := testArtifact()
		So(artifact, ShouldNotBeNil)

		id, err := artifact.Id()
		So(err, ShouldBeNil)
		So(id, ShouldNotBeEmpty)

		model, err := artifact.Model()
		So(err, ShouldBeNil)
		So(model, ShouldEqual, "test-model")

		messages, err := artifact.Messages()
		So(err, ShouldBeNil)
		So(messages.Len(), ShouldEqual, 0)

		tools, err := artifact.Tools()
		So(err, ShouldBeNil)
		So(tools.Len(), ShouldEqual, 0)

		process, err := artifact.Process()
		So(err, ShouldBeNil)
		So(string(process), ShouldEqual, "test-process")

		So(artifact.Temperature(), ShouldEqual, 0.7)
		So(artifact.TopP(), ShouldEqual, 1.0)
		So(artifact.TopK(), ShouldEqual, 40.0)
		So(artifact.PresencePenalty(), ShouldEqual, 0.0)
		So(artifact.FrequencyPenalty(), ShouldEqual, 0.0)
		So(artifact.MaxTokens(), ShouldEqual, 2048)
		So(artifact.Stream(), ShouldBeFalse)
	})
}
