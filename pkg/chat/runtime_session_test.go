package chat

import (
	"bytes"
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/compiler"
)

func TestSessionDrivesRuntimeGenerator(t *testing.T) {
	Convey("Given a Session whose generator is a RuntimeGenerator", t, func() {
		runtimeProgram, err := compiler.New(".").CompileBytes([]byte(deterministicChatYAML))
		So(err, ShouldBeNil)

		generator, err := NewRuntimeGenerator(RuntimeGeneratorOptions{
			Program:     runtimeProgram,
			Tokenizer:   tinyTokenizer{},
			GraphRunner: tinyModelRunner{vocabSize: 5},
		})
		So(err, ShouldBeNil)

		input := bytes.NewBufferString("hello\n/exit\n")
		output := &bytes.Buffer{}

		session := NewSession(context.Background(), input, output, generator, SessionConfig{
			Runtime: "runtime",
			Backend: "stub",
			Model:   "tiny",
		})

		Convey("Session.Run should drive the manifest and stream its emitted tokens", func() {
			// Session.Run takes the SessionRunner fast-path when the
			// generator implements it, handing the terminal streams
			// straight to the runtime program. Prompts/banners/etc.
			// belong to whichever manifest is loaded — the production
			// chat runtime emits "you> "/"caramba> " via io.emit_text
			// steps in chat.yml; the deterministicChatYAML fixture here
			// only emits sampled tokens, so that's what we verify.
			So(session.Run(), ShouldBeNil)
			text := output.String()
			So(text, ShouldContainSubstring, "<0><1><2><3><4>")
		})
	})
}
