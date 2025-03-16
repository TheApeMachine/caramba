package context

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestRead(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := testArtifact()

		Convey("When reading from the artifact", func() {
			// First get the expected marshaled data
			expected := artifact.Marshal()

			// Create a buffer of the right size
			p := make([]byte, len(expected))
			n, err := artifact.Read(p)

			So(err, ShouldEqual, io.EOF)
			So(n, ShouldEqual, len(expected))
			So(p, ShouldResemble, expected)
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given an empty artifact", t, func() {
		empty := &Artifact{}

		Convey("When writing a marshaled artifact", func() {
			artifact := testArtifact()

			// Get the marshaled data to write
			p := artifact.Marshal()

			// Write the marshaled data to the empty artifact
			n, err := empty.Write(p)

			So(err, ShouldBeNil)
			So(n, ShouldEqual, len(p))

			// Verify the empty artifact now matches the original
			emptyId, err := empty.Id()
			So(err, ShouldBeNil)
			originalId, _ := artifact.Id()
			So(emptyId, ShouldEqual, originalId)

			emptyModel, err := empty.Model()
			So(err, ShouldBeNil)
			originalModel, _ := artifact.Model()
			So(emptyModel, ShouldEqual, originalModel)

			emptyProcess, err := empty.Process()
			So(err, ShouldBeNil)
			originalProcess, _ := artifact.Process()
			So(emptyProcess, ShouldResemble, originalProcess)

			So(empty.Temperature(), ShouldEqual, artifact.Temperature())
			So(empty.TopP(), ShouldEqual, artifact.TopP())
			So(empty.TopK(), ShouldEqual, artifact.TopK())
			So(empty.PresencePenalty(), ShouldEqual, artifact.PresencePenalty())
			So(empty.FrequencyPenalty(), ShouldEqual, artifact.FrequencyPenalty())
			So(empty.MaxTokens(), ShouldEqual, artifact.MaxTokens())
			So(empty.Stream(), ShouldEqual, artifact.Stream())
		})

		Convey("When writing empty data", func() {
			n, err := empty.Write([]byte{})
			So(err, ShouldNotBeNil)
			So(n, ShouldEqual, 0)
		})
	})
}

func TestClose(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := testArtifact()

		Convey("When closing the artifact", func() {
			err := artifact.Close()
			So(err, ShouldBeNil)
		})
	})
}
