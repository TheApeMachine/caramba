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

		Convey("Session.Run should stream the assistant prompt, the tokens, and exit", func() {
			So(session.Run(), ShouldBeNil)
			text := output.String()
			So(text, ShouldContainSubstring, "you> ")
			So(text, ShouldContainSubstring, "caramba> ")
			So(text, ShouldContainSubstring, "<0><1><2><3><4>")
		})
	})
}
