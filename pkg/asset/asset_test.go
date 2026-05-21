package asset

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestWalk(t *testing.T) {
	Convey("Given embedded templates", t, func() {
		Convey("Walk", func() {
			Convey("It should return operation, block, and model schemas", func() {
				schemas, err := Walk("template/operation")

				So(err, ShouldBeNil)
				So(schemas, ShouldNotBeEmpty)

				blocks, err := Walk("template/block")

				So(err, ShouldBeNil)
				So(blocks, ShouldNotBeEmpty)

				models, err := Walk("template/model")

				So(err, ShouldBeNil)
				So(models, ShouldNotBeEmpty)
			})
		})
	})
}
