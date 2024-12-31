package ai

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/provider"
)

func TestNewContext(t *testing.T) {
	Convey("Given NewContext function", t, func() {
		system := &System{
			fragments: map[string]string{
				"prompt": "Test prompt",
				"name":   "TestBot",
				"role":   "assistant",
			},
		}
		params := &provider.GenerationParams{
			Thread: provider.NewThread(),
		}

		Convey("When creating a new context", func() {
			ctx := NewContext(system, params)

			Convey("Should initialize with correct values", func() {
				So(ctx.System, ShouldEqual, system)
				So(ctx.params, ShouldEqual, params)
				So(ctx.Thread, ShouldEqual, params.Thread)
				So(ctx.Scratchpad, ShouldNotBeNil)
				So(ctx.indent, ShouldEqual, 0)
			})
		})
	})
}

func TestContextCompile(t *testing.T) {
	Convey("Given Context Compile method", t, func() {
		system := &System{
			fragments: map[string]string{
				"prompt": "Test prompt",
				"name":   "TestBot",
				"role":   "assistant",
			},
		}
		params := &provider.GenerationParams{
			Thread: provider.NewThread(),
		}
		ctx := NewContext(system, params)

		Convey("When compiling empty scratchpad", func() {
			result := ctx.Compile()

			Convey("Should contain only system message", func() {
				So(result.Thread.Messages, ShouldHaveLength, 1)
				So(result.Thread.Messages[0].Role, ShouldEqual, provider.RoleSystem)
			})
		})

		Convey("When compiling with user and assistant messages", func() {
			ctx.Scratchpad.AddMessage(provider.NewMessage(provider.RoleUser, "User input"))
			ctx.Scratchpad.AddMessage(provider.NewMessage(provider.RoleAssistant, "Assistant response"))

			result := ctx.Compile()

			Convey("Should format messages correctly", func() {
				So(result.Thread.Messages, ShouldHaveLength, 3)
				So(result.Thread.Messages[1].Content, ShouldContainSubstring, "<goal>")
				So(result.Thread.Messages[2].Content, ShouldContainSubstring, "<response")
			})
		})
	})
}

func TestContextGetScratchpad(t *testing.T) {
	Convey("Given Context GetScratchpad method", t, func() {
		ctx := NewContext(&System{}, &provider.GenerationParams{})

		Convey("And the scratchpad is not empty", func() {
			ctx.Scratchpad.AddMessage(provider.NewMessage(provider.RoleAssistant, "Initial"))

			Convey("When getting a new scratchpad", func() {
				oldScratchpad := ctx.Scratchpad
				result := ctx.GetScratchpad()

				Convey("Should return context and create new scratchpad", func() {
					So(result, ShouldEqual, ctx)
					So(ctx.Scratchpad, ShouldNotBeNil)
					So(ctx.Scratchpad, ShouldNotEqual, oldScratchpad)
				})
			})
		})
	})
}

func TestContextAppend(t *testing.T) {
	Convey("Given Context Append method", t, func() {
		ctx := NewContext(&System{}, &provider.GenerationParams{})
		ctx.GetScratchpad()
		ctx.Scratchpad.AddMessage(provider.NewMessage(provider.RoleAssistant, "Initial"))

		Convey("When appending event text", func() {
			ctx.Append(provider.Event{Text: " additional"})

			Convey("Should append to last message", func() {
				So(ctx.Scratchpad.Messages[0].Content, ShouldEqual, "Initial additional")
			})
		})
	})
}

func TestContextToolCall(t *testing.T) {
	Convey("Given Context ToolCall method", t, func() {
		ctx := NewContext(&System{}, &provider.GenerationParams{})
		ctx.GetScratchpad()

		Convey("When handling tool call event", func() {
			ctx.ToolCall(provider.Event{Text: "tool result"})

			Convey("Should add tool message with correct formatting", func() {
				So(ctx.Scratchpad.Messages, ShouldHaveLength, 1)
				So(ctx.Scratchpad.Messages[0].Role, ShouldEqual, provider.RoleTool)
				So(ctx.Scratchpad.Messages[0].Content, ShouldContainSubstring, "<tool>")
				So(ctx.Scratchpad.Messages[0].Content, ShouldContainSubstring, "tool result")
			})
		})
	})
}

func TestContextDone(t *testing.T) {
	Convey("Given Context Done method", t, func() {
		ctx := NewContext(&System{}, &provider.GenerationParams{})
		ctx.GetScratchpad()

		Convey("When checking completion without [STOP]", func() {
			ctx.Scratchpad.AddMessage(provider.NewMessage(provider.RoleAssistant, "Normal message"))

			Convey("Should return false", func() {
				So(ctx.Done(), ShouldBeFalse)
			})
		})

		Convey("When checking completion with [STOP]", func() {
			ctx.Scratchpad.AddMessage(provider.NewMessage(provider.RoleAssistant, "Message with [STOP]"))

			Convey("Should return true", func() {
				So(ctx.Done(), ShouldBeTrue)
			})
		})
	})
}
