package event

import (
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestRead(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := testArtifact()

		Convey("When the artifact is read", func() {
			// First get the expected marshaled data
			expected, err := artifact.Message().Marshal()
			So(err, ShouldBeNil)

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
			p, err := artifact.Message().Marshal()
			So(err, ShouldBeNil)

			// Write the marshaled data to the empty artifact
			n, err := empty.Write(p)

			So(err, ShouldBeNil)
			So(n, ShouldEqual, len(p))

			// Verify the empty artifact now matches the original
			emptyMarshaled, err := empty.Message().Marshal()
			So(err, ShouldBeNil)
			So(emptyMarshaled, ShouldResemble, p)
		})
	})
}
