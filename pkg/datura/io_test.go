package datura

import (
	"bytes"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func testArtifact() *Artifact {
	return New(
		WithMediatype(MediaTypeCapnp),
		WithRole(ArtifactRoleUser),
		WithScope(ArtifactScopeContext),
		WithEncryptedPayload([]byte("test payload")),
	)
}

func TestRead(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := testArtifact()

		Convey("When the artifact is read", func() {
			// First get the expected marshaled data
			expected, err := artifact.Message().Marshal()
			So(err, ShouldBeNil)

			// Create a buffer of the right size
			buf := bytes.NewBuffer([]byte{})
			n, err := io.Copy(buf, artifact)

			So(err, ShouldBeNil)
			So(n, ShouldEqual, len(expected))
			So(buf.Bytes(), ShouldResemble, expected)
		})
	})
}

func TestWrite(t *testing.T) {
	Convey("Given an empty artifact", t, func() {
		newArtifact := New()

		Convey("When writing a marshaled artifact", func() {
			artifact := testArtifact()

			original, err := artifact.Payload()
			So(err, ShouldBeNil)

			// Get the expected marshaled length
			expectedMarshaled, err := artifact.Message().Marshal()
			So(err, ShouldBeNil)
			expectedLen := len(expectedMarshaled)

			// Get the marshaled data to write
			n, err := io.Copy(newArtifact, artifact)

			So(err, ShouldBeNil)
			// Assert that the number of bytes written equals the marshaled length
			So(n, ShouldEqual, expectedLen)

			payload, err := newArtifact.Payload()
			So(err, ShouldBeNil)

			So(payload, ShouldEqual, original)
		})
	})
}
