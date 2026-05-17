package executor

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestScope(t *testing.T) {
	Convey("Given a root scope", t, func() {
		root := NewScope()

		Convey("Get on a missing name should error", func() {
			_, err := root.Get("missing")
			So(err, ShouldNotBeNil)
		})

		Convey("Set then Get should return the bound value", func() {
			root.Set("user_text", "hello")
			value, err := root.Get("user_text")
			So(err, ShouldBeNil)
			So(value, ShouldEqual, "hello")
		})

		Convey("A child scope should see parent bindings", func() {
			root.Set("history", []int{1, 2, 3})
			child := root.Child()
			value, err := child.Get("history")
			So(err, ShouldBeNil)
			So(value, ShouldResemble, []int{1, 2, 3})
		})

		Convey("A child should shadow a parent binding without mutating it", func() {
			root.Set("position", 7)
			child := root.Child()
			child.Set("position", 42)

			fromChild, _ := child.Get("position")
			fromRoot, _ := root.Get("position")

			So(fromChild, ShouldEqual, 42)
			So(fromRoot, ShouldEqual, 7)
		})

		Convey("Has should walk the parent chain", func() {
			root.Set("seed", 12345)
			child := root.Child()
			So(child.Has("seed"), ShouldBeTrue)
			So(child.Has("missing"), ShouldBeFalse)
		})
	})
}
