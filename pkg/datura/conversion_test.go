package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

type TestStruct struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

func TestBytes(t *testing.T) {
	Convey("Given an artifact", t, func() {
		testData := New(
			WithMediatype(MediaTypeApplicationJson),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeContext),
		)

		Convey("When converting to bytes", func() {
			buf := testData.Bytes()

			Convey("Then it should unmarshal correctly", func() {
				So(buf, ShouldNotBeEmpty)
			})

			Convey("When converting from bytes", func() {
				artifact := New(
					WithBytes(buf),
				)

				Convey("Then it should unmarshal correctly", func() {
					So(artifact, ShouldResemble, testData)
				})
			})
		})
	})
}
