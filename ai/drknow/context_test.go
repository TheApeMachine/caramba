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
		ctx := NewContext(identity)

		Convey("Then it should be properly initialized", func() {
			So(ctx, ShouldNotBeNil)
			So(ctx.Identity, ShouldEqual, identity)
			So(ctx.Toolcalls, ShouldBeNil)
			So(ctx.indent, ShouldEqual, 0)
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
		ctx := NewContext(identity)

		Convey("When compiling with cycle information", func() {
			params := ctx.Compile(1, 5)

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
		ctx := NewContext(identity)

		Convey("When getting string with system messages", func() {
			result := ctx.String(true)

			Convey("Then it should include system messages", func() {
				So(result, ShouldContainSubstring, "system")
				So(result, ShouldContainSubstring, "test system")
			})
		})

		Convey("When getting string without system messages", func() {
			result := ctx.String(false)

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
		ctx := NewContext(identity)

		Convey("When resetting the context", func() {
			// Add some messages and tool calls
			ctx.AddMessage(provider.NewMessage(provider.RoleUser, "test message"))

			// Create an event that implements the Event interface
			event := provider.NewEvent("generate:toolcall", provider.EventFunction, "Tool called", "", nil)
			ctx.Toolcalls = append(ctx.Toolcalls, event)
			ctx.Reset()

			Convey("Then it should clear messages except system", func() {
				So(len(ctx.Identity.Params.Thread.Messages), ShouldEqual, 1)
				So(ctx.Identity.Params.Thread.Messages[0].Role, ShouldEqual, provider.RoleSystem)
			})

			Convey("Then it should clear tool calls", func() {
				So(len(ctx.Toolcalls), ShouldEqual, 0)
			})
		})
	})
}

func TestAddMessage(t *testing.T) {
	Convey("Given a Context instance", t, func() {
		identity := NewIdentity(context.TODO(), "test", "test system")
		ctx := NewContext(identity)

		Convey("When adding a message", func() {
			msg := provider.NewMessage(provider.RoleUser, "test message")
			ctx.AddMessage(msg)

			Convey("Then it should be added to the thread", func() {
				So(len(ctx.Identity.Params.Thread.Messages), ShouldBeGreaterThan, 1)
				So(ctx.Identity.Params.Thread.Messages[len(ctx.Identity.Params.Thread.Messages)-1], ShouldEqual, msg)
			})
		})
	})
}
