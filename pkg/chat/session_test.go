package chat

import (
	"bytes"
	"context"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSession_Run(test *testing.T) {
	Convey("Given an interactive chat session", test, func() {
		input := strings.NewReader("hello\n/help\n/exit\n")
		output := bytes.NewBuffer(nil)
		generator := &recordingGenerator{chunks: []string{"hi", " there"}}
		session := NewSession(
			context.Background(),
			input,
			output,
			generator,
			SessionConfig{Runtime: "test", ShowBanner: true},
		)

		Convey("It should stream responses until exit", func() {
			err := session.Run()

			So(err, ShouldBeNil)
			So(output.String(), ShouldContainSubstring, "caramba chat runtime=test")
			So(output.String(), ShouldContainSubstring, "you> caramba> hi there")
			So(output.String(), ShouldContainSubstring, "commands: /help /exit /quit")
			So(generator.prompts, ShouldResemble, []string{"hello"})
		})
	})
}

func TestSession_RunPrompt(test *testing.T) {
	Convey("Given a one-shot chat session", test, func() {
		output := bytes.NewBuffer(nil)
		generator := &recordingGenerator{chunks: []string{"streamed"}}
		session := NewSession(
			context.Background(),
			strings.NewReader(""),
			output,
			generator,
			SessionConfig{Runtime: "test"},
		)

		Convey("It should stream one response", func() {
			err := session.RunPrompt("hello")

			So(err, ShouldBeNil)
			So(output.String(), ShouldEqual, "caramba> streamed\n")
			So(generator.prompts, ShouldResemble, []string{"hello"})
		})

		Convey("It should include backend in the banner when configured", func() {
			output := bytes.NewBuffer(nil)
			session := NewSession(
				context.Background(),
				strings.NewReader(""),
				output,
				generator,
				SessionConfig{
					Runtime:    "model",
					Backend:    "metal",
					Model:      "openai-community/gpt2",
					ShowBanner: true,
				},
			)

			err := session.RunPrompt("hello")

			So(err, ShouldBeNil)
			So(output.String(), ShouldContainSubstring, "backend=metal")
		})
	})
}

func BenchmarkSession_RunPrompt(benchmark *testing.B) {
	generator := &recordingGenerator{chunks: []string{"ok"}}

	for benchmark.Loop() {
		output := bytes.NewBuffer(nil)
		session := NewSession(
			context.Background(),
			strings.NewReader(""),
			output,
			generator,
			SessionConfig{Runtime: "bench"},
		)
		_ = session.RunPrompt("hello")
	}
}

type recordingGenerator struct {
	prompts []string
	chunks  []string
}

func (generator *recordingGenerator) Generate(
	_ context.Context,
	prompt string,
	emit func(string) error,
) error {
	generator.prompts = append(generator.prompts, prompt)

	for _, chunk := range generator.chunks {
		if err := emit(chunk); err != nil {
			return err
		}
	}

	return nil
}
