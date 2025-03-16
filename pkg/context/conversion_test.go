package context

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

				newModel, err := newArtifact.Model()
				So(err, ShouldBeNil)
				originalModel, _ := original.Model()
				So(newModel, ShouldEqual, originalModel)

				newProcess, err := newArtifact.Process()
				So(err, ShouldBeNil)
				originalProcess, _ := original.Process()
				So(newProcess, ShouldResemble, originalProcess)

				So(newArtifact.Temperature(), ShouldEqual, original.Temperature())
				So(newArtifact.TopP(), ShouldEqual, original.TopP())
				So(newArtifact.TopK(), ShouldEqual, original.TopK())
				So(newArtifact.PresencePenalty(), ShouldEqual, original.PresencePenalty())
				So(newArtifact.FrequencyPenalty(), ShouldEqual, original.FrequencyPenalty())
				So(newArtifact.MaxTokens(), ShouldEqual, original.MaxTokens())
				So(newArtifact.Stream(), ShouldEqual, original.Stream())
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

				newModel, err := newArtifact.Model()
				So(err, ShouldBeNil)
				originalModel, _ := original.Model()
				So(newModel, ShouldEqual, originalModel)

				newProcess, err := newArtifact.Process()
				So(err, ShouldBeNil)
				originalProcess, _ := original.Process()
				So(newProcess, ShouldResemble, originalProcess)

				So(newArtifact.Temperature(), ShouldEqual, original.Temperature())
				So(newArtifact.TopP(), ShouldEqual, original.TopP())
				So(newArtifact.TopK(), ShouldEqual, original.TopK())
				So(newArtifact.PresencePenalty(), ShouldEqual, original.PresencePenalty())
				So(newArtifact.FrequencyPenalty(), ShouldEqual, original.FrequencyPenalty())
				So(newArtifact.MaxTokens(), ShouldEqual, original.MaxTokens())
				So(newArtifact.Stream(), ShouldEqual, original.Stream())
			})
		})

		Convey("When unpacking empty data", func() {
			emptyArtifact := &Artifact{}
			result := emptyArtifact.Unpack([]byte{})
			So(result, ShouldBeNil)
		})
	})
}
