package ai

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
)

func TestNewInterpreter(t *testing.T) {
	Convey("Given a call to NewInterpreter", t, func() {
		ctx := drknow.QuickContext("test system")
		interpreter := NewInterpreter(ctx)

		Convey("Then it should be properly initialized", func() {
			So(interpreter, ShouldNotBeNil)
			So(interpreter.ctx, ShouldEqual, ctx)
			So(interpreter.commands, ShouldNotBeNil)
			So(len(interpreter.commands), ShouldEqual, 0)
		})
	})
}

func TestInterpret(t *testing.T) {
	Convey("Given an Interpreter instance", t, func() {
		ctx := drknow.QuickContext("test system")
		interpreter := NewInterpreter(ctx)

		Convey("When interpreting with no messages", func() {
			state, _ := interpreter.Interpret()

			Convey("Then it should return a state without changes", func() {
				So(state, ShouldNotBeNil)
				So(len(interpreter.commands), ShouldEqual, 0)
			})
		})

		Convey("When interpreting a non-assistant message", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleUser, "<help>"))
			state, _ := interpreter.Interpret()

			Convey("Then it should return a state without changes", func() {
				So(state, ShouldNotBeNil)
				So(len(interpreter.commands), ShouldEqual, 0)
			})
		})

		Convey("When interpreting a valid command", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<help>"))
			state, _ := interpreter.Interpret()

			Convey("Then it should parse the command", func() {
				So(state, ShouldNotBeNil)
				So(len(interpreter.commands), ShouldEqual, 1)
				So(interpreter.commands[0].Task, ShouldHaveSameTypeAs, taskMap["help"])
				So(len(interpreter.commands[0].Args), ShouldEqual, 0)
			})
		})

		Convey("When interpreting a command with parameters", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, `<web url="https://example.com">`))
			state, _ := interpreter.Interpret()

			Convey("Then it should parse the command and parameters", func() {
				So(state, ShouldNotBeNil)
				So(len(interpreter.commands), ShouldEqual, 1)
				So(interpreter.commands[0].Task, ShouldHaveSameTypeAs, taskMap["web"])
				So(interpreter.commands[0].Args["url"], ShouldEqual, "https://example.com")
			})
		})

		Convey("When interpreting an unknown command", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<unknowncommand>"))
			state, _ := interpreter.Interpret()

			Convey("Then it should ignore the unknown command", func() {
				So(state, ShouldNotBeNil)
				So(len(interpreter.commands), ShouldEqual, 0)
			})
		})

		Convey("When interpreting multiple commands", func() {
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "<help><web url=\"https://example.com\">"))
			state, _ := interpreter.Interpret()

			Convey("Then it should parse all commands", func() {
				So(state, ShouldNotBeNil)
				So(len(interpreter.commands), ShouldEqual, 2)
				So(interpreter.commands[0].Task, ShouldHaveSameTypeAs, taskMap["help"])
				So(interpreter.commands[1].Task, ShouldHaveSameTypeAs, taskMap["web"])
				So(interpreter.commands[1].Args["url"], ShouldEqual, "https://example.com")
			})
		})
	})
}

func TestExecute(t *testing.T) {
	Convey("Given an Interpreter instance", t, func() {
		ctx := drknow.QuickContext("test system")
		interpreter := NewInterpreter(ctx)

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
