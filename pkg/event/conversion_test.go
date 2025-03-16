package event

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMarshal(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := testArtifact()

		Convey("When marshaling the artifact", func() {
			// Get expected data directly from Message().Marshal()
			expected, err := artifact.Message().Marshal()
			So(err, ShouldBeNil)

			// Create buffer and marshal into it
			p := make([]byte, len(expected))
			artifact.Marshal(p)

			So(p, ShouldResemble, expected)
		})
	})
}

func TestUnmarshal(t *testing.T) {
	Convey("Given a marshaled artifact", t, func() {
		original := testArtifact()

		// Marshal the original artifact
		data, err := original.Message().Marshal()
		So(err, ShouldBeNil)

		Convey("When unmarshaling into a new artifact", func() {
			// Create empty artifact and unmarshal into it
			newArtifact := &Artifact{}
			result := newArtifact.Unmarshal(data)

			So(result, ShouldNotBeNil)

			// Verify the unmarshaled artifact matches the original
			newData, err := newArtifact.Message().Marshal()
			So(err, ShouldBeNil)
			So(newData, ShouldResemble, data)

			// Verify some specific fields to ensure proper unmarshaling
			originalPayload, err := original.Payload()
			So(err, ShouldBeNil)
			newPayload, err := newArtifact.Payload()
			So(err, ShouldBeNil)
			So(newPayload, ShouldResemble, originalPayload)

			originalType, err := original.Type()
			So(err, ShouldBeNil)
			newType, err := newArtifact.Type()
			So(err, ShouldBeNil)
			So(newType, ShouldEqual, originalType)
		})

		Convey("When unmarshaling empty data", func() {
			emptyArtifact := &Artifact{}
			result := emptyArtifact.Unmarshal([]byte{})
			So(result, ShouldBeNil)
		})
	})
}
