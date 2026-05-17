package program

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestParseValueRef(t *testing.T) {
	Convey("Given the runtime value-ref parser", t, func() {
		Convey("It should treat a bare identifier as a local value", func() {
			ref, err := ParseValueRef("user_text")
			So(err, ShouldBeNil)
			So(ref.Namespace, ShouldEqual, NamespaceLocal)
			So(ref.Name, ShouldEqual, "user_text")
			So(ref.Path, ShouldBeEmpty)
		})

		Convey("It should split namespace and name on a known prefix", func() {
			ref, err := ParseValueRef("state.history")
			So(err, ShouldBeNil)
			So(ref.Namespace, ShouldEqual, NamespaceState)
			So(ref.Name, ShouldEqual, "history")
		})

		Convey("It should capture trailing path segments", func() {
			ref, err := ParseValueRef("sampler.main.stop_matched")
			So(err, ShouldBeNil)
			So(ref.Namespace, ShouldEqual, NamespaceSampler)
			So(ref.Name, ShouldEqual, "main")
			So(ref.Path, ShouldResemble, []string{"stop_matched"})
		})

		Convey("It should leave unknown prefixes as local dotted names", func() {
			ref, err := ParseValueRef("history.length")
			So(err, ShouldBeNil)
			So(ref.Namespace, ShouldEqual, NamespaceLocal)
			So(ref.Name, ShouldEqual, "history.length")
		})

		Convey("It should reject an empty input", func() {
			_, err := ParseValueRef("")
			So(err, ShouldNotBeNil)
		})

		Convey("It should round-trip through String", func() {
			cases := []string{
				"user_text",
				"state.history",
				"sampler.main.stop_matched",
			}

			for _, raw := range cases {
				ref, err := ParseValueRef(raw)
				So(err, ShouldBeNil)
				So(ref.String(), ShouldEqual, raw)
			}
		})

		Convey("It should report equality only when namespace, name and path match", func() {
			lhs, _ := ParseValueRef("state.history")
			rhs, _ := ParseValueRef("state.history")
			other, _ := ParseValueRef("state.position")

			So(lhs.Equal(rhs), ShouldBeTrue)
			So(lhs.Equal(other), ShouldBeFalse)
		})
	})
}
