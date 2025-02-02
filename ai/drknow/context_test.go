package drknow

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
)

func TestNewContext(t *testing.T) {
	Convey("Given a call to NewContext", t, func() {
		identity := NewIdentity(context.TODO(), "test", "test system")
		ctx, cancel := context.WithCancel(context.Background())
		dctx := NewContext(identity, ctx, cancel)

		Convey("Then it should be properly initialized", func() {
			So(dctx, ShouldNotBeNil)
			So(dctx.Identity, ShouldEqual, identity)
		})
	})
}

func TestQuickContext(t *testing.T) {
	Convey("Given a call to QuickContext", t, func() {
		ctx := QuickContext("test system", "test_addition")

		Convey("Then it should be properly initialized", func() {
			So(ctx, ShouldNotBeNil)
			So(ctx.Identity, ShouldNotBeNil)
			So(ctx.Identity.Role, ShouldEqual, "reasoner")
		})
	})
}

func TestCompile(t *testing.T) {
	Convey("Given a Context instance", t, func() {
		identity := NewIdentity(context.TODO(), "test", "test system")
		ctx, cancel := context.WithCancel(context.Background())
		dctx := NewContext(identity, ctx, cancel)

		Convey("When compiling with cycle information", func() {
			params := dctx.Compile()

			Convey("Then it should return valid generation params", func() {
				So(params, ShouldNotBeNil)
				So(params.Thread, ShouldNotBeNil)
				So(len(params.Thread.Messages), ShouldBeGreaterThan, 0)
			})
		})
	})
}

func TestString(t *testing.T) {
	Convey("Given a Context instance", t, func() {
		identity := NewIdentity(context.TODO(), "test", "test system")
		ctx, cancel := context.WithCancel(context.Background())
		dctx := NewContext(identity, ctx, cancel)

		Convey("When getting string with system messages", func() {
			result := dctx.String(true)

			Convey("Then it should include system messages", func() {
				So(result, ShouldContainSubstring, "system")
				So(result, ShouldContainSubstring, "test system")
			})
		})

		Convey("When getting string without system messages", func() {
			result := dctx.String(false)

			Convey("Then it should not include system messages", func() {
				So(result, ShouldNotContainSubstring, "system")
				So(result, ShouldNotContainSubstring, "test system")
			})
		})
	})
}

func TestReset(t *testing.T) {
	Convey("Given a Context instance", t, func() {
		identity := NewIdentity(context.TODO(), "test", "test system")
		ctx, cancel := context.WithCancel(context.Background())
		dctx := NewContext(identity, ctx, cancel)

		Convey("When resetting the context", func() {
			// Add some messages and tool calls
			dctx.AddIteration("test message")

			// Create an event that implements the Event interface
			dctx.Reset()

			Convey("Then it should clear messages except system", func() {
				So(len(dctx.Identity.Params.Thread.Messages), ShouldEqual, 1)
				So(dctx.Identity.Params.Thread.Messages[0].Role, ShouldEqual, provider.RoleSystem)
			})
		})
	})
}

func TestAddMessage(t *testing.T) {
	Convey("Given a Context instance", t, func() {
		identity := NewIdentity(context.TODO(), "test", "test system")
		ctx, cancel := context.WithCancel(context.Background())
		dctx := NewContext(identity, ctx, cancel)

		Convey("When adding a message", func() {
			dctx.AddIteration("test message")

			Convey("Then it should be added to the thread", func() {
				So(len(dctx.Identity.Params.Thread.Messages), ShouldBeGreaterThan, 1)
				So(dctx.Identity.Params.Thread.Messages[len(dctx.Identity.Params.Thread.Messages)-1], ShouldEqual, "test message")
			})
		})
	})
}
