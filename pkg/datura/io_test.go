package datura

import (
	"bytes"
	"io"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func testArtifact() *ArtifactBuilder {
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
			expected, err := artifact.Artifact.Message().Marshal()
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

			// Get the marshaled data to write
			_, err = io.Copy(newArtifact, artifact)

			So(err, ShouldBeNil)

			payload, err := newArtifact.Payload()
			So(err, ShouldBeNil)

			So(payload, ShouldResemble, original)
		})
	})
}
