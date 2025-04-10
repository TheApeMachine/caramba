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
			WithPayload([]byte("Hello, world!")),
			WithMeta("test", "test"),
		)

		payload, err := testData.Payload()
		So(err, ShouldBeNil)
		So(string(payload), ShouldEqual, "Hello, world!")

		Convey("When converting to bytes", func() {
			buf := testData.Bytes()

			Convey("Then it should unmarshal correctly", func() {
				So(buf, ShouldNotBeEmpty)
			})

			Convey("And it should have the same payload", func() {
				artifact := New(
					WithBytes(buf),
				)

				payload, err := artifact.Payload()
				So(err, ShouldBeNil)
				So(string(payload), ShouldEqual, "Hello, world!")
			})

			Convey("And it should have the same metadata", func() {
				artifact := New(
					WithBytes(buf),
				)

				meta := GetMetaValue[string](artifact, "test")

				So(meta, ShouldEqual, "test")
			})
		})
	})
}
