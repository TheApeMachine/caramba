package message

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func testArtifact() *Artifact {
	return New(
		UserRole,
		"test-name",
		"test-content",
	)
}

func TestNewArtifact(t *testing.T) {
	Convey("Given a new artifact", t, func() {
		artifact := testArtifact()
		So(artifact, ShouldNotBeNil)

		id, err := artifact.Id()
		So(err, ShouldBeNil)
		So(id, ShouldNotBeEmpty)

		name, err := artifact.Name()
		So(err, ShouldBeNil)
		So(name, ShouldEqual, "test-name")

		role, err := artifact.Role()
		So(err, ShouldBeNil)
		So(role, ShouldEqual, UserRole.String())

		content, err := artifact.Content()
		So(err, ShouldBeNil)
		So(content, ShouldEqual, "test-content")
	})
}
