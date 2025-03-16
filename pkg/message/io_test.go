package message

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
			emptyId, err := empty.Id()
			So(err, ShouldBeNil)
			originalId, _ := artifact.Id()
			So(emptyId, ShouldEqual, originalId)

			emptyRole, err := empty.Role()
			So(err, ShouldBeNil)
			originalRole, _ := artifact.Role()
			So(emptyRole, ShouldEqual, originalRole)

			emptyName, err := empty.Name()
			So(err, ShouldBeNil)
			originalName, _ := artifact.Name()
			So(emptyName, ShouldEqual, originalName)

			emptyContent, err := empty.Content()
			So(err, ShouldBeNil)
			originalContent, _ := artifact.Content()
			So(emptyContent, ShouldEqual, originalContent)
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
