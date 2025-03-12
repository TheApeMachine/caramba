package ai

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

// TestNewContext tests the NewContext constructor
func TestNewContext(t *testing.T) {
	Convey("Given no parameters", t, func() {
		Convey("When creating a new Context", func() {
			ctx := NewContext()

			Convey("Then the context should have the correct properties", func() {
				So(ctx, ShouldNotBeNil)
				So(ctx.ContextData, ShouldNotBeNil)
				So(ctx.Messages, ShouldNotBeNil)
				So(ctx.Tools, ShouldNotBeNil)
				So(ctx.Buffer, ShouldNotBeNil)
			})
		})
	})
}
