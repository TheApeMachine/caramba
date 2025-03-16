package event

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func testArtifact() *Artifact {
	return New(
		"test",
		MessageEvent,
		UserRole,
		[]byte("test"),
	)
}

func TestNewArtifact(t *testing.T) {
	Convey("Given a new artifact", t, func() {
		artifact := testArtifact()
		So(artifact, ShouldNotBeNil)

		payload, err := artifact.Payload()
		So(err, ShouldBeNil)
		So(payload, ShouldResemble, []byte("test"))

		role, err := artifact.Role()
		So(err, ShouldBeNil)
		So(role, ShouldEqual, UserRole.String())

		origin, err := artifact.Origin()
		So(err, ShouldBeNil)
		So(origin, ShouldEqual, "test")

		typ, err := artifact.Type()
		So(err, ShouldBeNil)
		So(typ, ShouldEqual, MessageEvent.String())
	})
}
