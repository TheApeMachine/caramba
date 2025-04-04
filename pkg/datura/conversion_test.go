package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type TestStruct struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

func TestTo(t *testing.T) {
	Convey("Given an artifact with JSON payload", t, func() {
		testData := TestStruct{
			Name:  "test",
			Value: 42,
		}

		artifact := New(
			WithMediatype(MediaTypeApplicationJson),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
		)

		err := artifact.From(testData)
		So(err, ShouldBeNil)

		Convey("When converting to a struct", func() {
			var result TestStruct
			err := artifact.To(&result)

			Convey("Then it should unmarshal correctly", func() {
				So(err, ShouldBeNil)
				So(result.Name, ShouldEqual, testData.Name)
				So(result.Value, ShouldEqual, testData.Value)
			})
		})
	})
}

func TestFrom(t *testing.T) {
	Convey("Given a struct with data", t, func() {
		testData := TestStruct{
			Name:  "test",
			Value: 42,
		}

		artifact := New(
			WithMediatype(MediaTypeApplicationJson),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
		)

		Convey("When converting from the struct", func() {
			err := artifact.From(testData)

			Convey("Then it should marshal correctly", func() {
				So(err, ShouldBeNil)

				payload, err := artifact.DecryptPayload()
				So(err, ShouldBeNil)
				So(payload, ShouldNotBeEmpty)

				var result TestStruct
				err = artifact.To(&result)
				So(err, ShouldBeNil)
				So(result.Name, ShouldEqual, testData.Name)
				So(result.Value, ShouldEqual, testData.Value)
			})
		})
	})
}

func TestUnmarshal(t *testing.T) {
	Convey("Given a marshaled artifact", t, func() {
		original := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
			WithPayload([]byte("test payload")),
		)

		marshaled, err := original.Message().Marshal()
		So(err, ShouldBeNil)

		Convey("When unmarshaling the bytes", func() {
			result := Unmarshal(marshaled)

			Convey("Then it should recreate the artifact correctly", func() {
				So(result, ShouldNotBeNil)

				originalPayload, err := original.DecryptPayload()
				So(err, ShouldBeNil)

				resultPayload, err := result.DecryptPayload()
				So(err, ShouldBeNil)

				So(resultPayload, ShouldResemble, originalPayload)
			})
		})
	})
}

func TestError(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
		)

		testError := errnie.New(errnie.WithMessage("test error"))

		Convey("When setting an error", func() {
			err := artifact.Error(testError)

			Convey("Then it should return the error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "test error")
			})
		})

		Convey("When setting an empty error", func() {
			emptyError := errnie.New(errnie.WithMessage(""))
			err := artifact.Error(emptyError)

			Convey("Then it should handle empty error gracefully", func() {
				So(err, ShouldNotBeNil)
			})
		})
	})
}
