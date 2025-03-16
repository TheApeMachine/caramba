package message

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMarshalUnmarshal(t *testing.T) {
	Convey("Given an artifact", t, func() {
		original := testArtifact()

		Convey("When marshaling the artifact", func() {
			data := original.Marshal()
			So(data, ShouldNotBeNil)
			So(len(data), ShouldBeGreaterThan, 0)

			Convey("And unmarshaling back into a new artifact", func() {
				newArtifact := &Artifact{}
				result := newArtifact.Unmarshal(data)

				So(result, ShouldNotBeNil)

				// Verify fields match
				newId, err := newArtifact.Id()
				So(err, ShouldBeNil)
				originalId, _ := original.Id()
				So(newId, ShouldEqual, originalId)

				newRole, err := newArtifact.Role()
				So(err, ShouldBeNil)
				originalRole, _ := original.Role()
				So(newRole, ShouldEqual, originalRole)

				newName, err := newArtifact.Name()
				So(err, ShouldBeNil)
				originalName, _ := original.Name()
				So(newName, ShouldEqual, originalName)

				newContent, err := newArtifact.Content()
				So(err, ShouldBeNil)
				originalContent, _ := original.Content()
				So(newContent, ShouldEqual, originalContent)
			})
		})

		Convey("When unmarshaling empty data", func() {
			emptyArtifact := &Artifact{}
			result := emptyArtifact.Unmarshal([]byte{})
			So(result, ShouldBeNil)
		})
	})
}

func TestPackUnpack(t *testing.T) {
	Convey("Given an artifact", t, func() {
		original := testArtifact()

		Convey("When packing the artifact", func() {
			data := original.Pack()
			So(data, ShouldNotBeNil)
			So(len(data), ShouldBeGreaterThan, 0)

			Convey("And unpacking back into a new artifact", func() {
				newArtifact := &Artifact{}
				result := newArtifact.Unpack(data)

				So(result, ShouldNotBeNil)

				// Verify fields match
				newId, err := newArtifact.Id()
				So(err, ShouldBeNil)
				originalId, _ := original.Id()
				So(newId, ShouldEqual, originalId)

				newRole, err := newArtifact.Role()
				So(err, ShouldBeNil)
				originalRole, _ := original.Role()
				So(newRole, ShouldEqual, originalRole)

				newName, err := newArtifact.Name()
				So(err, ShouldBeNil)
				originalName, _ := original.Name()
				So(newName, ShouldEqual, originalName)

				newContent, err := newArtifact.Content()
				So(err, ShouldBeNil)
				originalContent, _ := original.Content()
				So(newContent, ShouldEqual, originalContent)
			})
		})

		Convey("When unpacking empty data", func() {
			emptyArtifact := &Artifact{}
			result := emptyArtifact.Unpack([]byte{})
			So(result, ShouldBeNil)
		})
	})
}
