package ai

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

func TestNewInterpreter(t *testing.T) {
	Convey("Given a call to NewInterpreter", t, func() {
		ctx := drknow.QuickContext("test system")
		accumulator := stream.NewAccumulator()
		interpreter := NewInterpreter(ctx, accumulator)

		Convey("Then it should be properly initialized", func() {
			So(interpreter, ShouldNotBeNil)
			So(interpreter.ctx, ShouldEqual, ctx)
			So(interpreter.accumulator, ShouldEqual, accumulator)
			So(interpreter.commands, ShouldNotBeNil)
			So(len(interpreter.commands), ShouldEqual, 0)
		})
	})
}

func TestInterpret(t *testing.T) {
	Convey("Given an Interpreter instance", t, func() {
		ctx := drknow.QuickContext("test system")
		accumulator := stream.NewAccumulator()
		interpreter := NewInterpreter(ctx, accumulator)

		Convey("When interpreting with no messages", func() {
			result := interpreter.Interpret()

			Convey("Then it should return itself without changes", func() {
				So(result, ShouldEqual, interpreter)
				So(len(result.commands), ShouldEqual, 0)
			})
		})

		Convey("When interpreting a non-assistant message", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleUser, "<help>"))
			result := interpreter.Interpret()

			Convey("Then it should return itself without changes", func() {
				So(result, ShouldEqual, interpreter)
				So(len(result.commands), ShouldEqual, 0)
			})
		})

		Convey("When interpreting a valid command", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<help>"))
			result := interpreter.Interpret()

			Convey("Then it should parse the command", func() {
				So(result, ShouldEqual, interpreter)
				So(len(result.commands), ShouldEqual, 1)
				So(result.commands[0].Task, ShouldHaveSameTypeAs, taskMap["help"])
				So(len(result.commands[0].Args), ShouldEqual, 0)
			})
		})

		Convey("When interpreting a command with parameters", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, `<web url="https://example.com">`))
			result := interpreter.Interpret()

			Convey("Then it should parse the command and parameters", func() {
				So(result, ShouldEqual, interpreter)
				So(len(result.commands), ShouldEqual, 1)
				So(result.commands[0].Task, ShouldHaveSameTypeAs, taskMap["web"])
				So(result.commands[0].Args["url"], ShouldEqual, "https://example.com")
			})
		})

		Convey("When interpreting an unknown command", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<unknowncommand>"))
			result := interpreter.Interpret()

			Convey("Then it should ignore the unknown command", func() {
				So(result, ShouldEqual, interpreter)
				So(len(result.commands), ShouldEqual, 0)
			})
		})

		Convey("When interpreting multiple commands", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<help><web url=\"https://example.com\">"))
			result := interpreter.Interpret()

			Convey("Then it should parse all commands", func() {
				So(result, ShouldEqual, interpreter)
				So(len(result.commands), ShouldEqual, 2)
				So(result.commands[0].Task, ShouldHaveSameTypeAs, taskMap["help"])
				So(result.commands[1].Task, ShouldHaveSameTypeAs, taskMap["web"])
				So(result.commands[1].Args["url"], ShouldEqual, "https://example.com")
			})
		})
	})
}

func TestExecute(t *testing.T) {
	Convey("Given an Interpreter instance", t, func() {
		ctx := drknow.QuickContext("test system")
		accumulator := stream.NewAccumulator()
		interpreter := NewInterpreter(ctx, accumulator)

		Convey("When executing with no commands", func() {
			interpreter.Execute()

			Convey("Then it should complete without error", func() {
				So(len(interpreter.commands), ShouldEqual, 0)
			})
		})

		Convey("When executing commands", func() {
			// Add a test command (using ignore as it's a no-op)
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<ignore>"))
			interpreter.Interpret()
			interpreter.Execute()

			Convey("Then it should execute all commands", func() {
				So(len(interpreter.commands), ShouldEqual, 1)
			})
		})
	})
}
