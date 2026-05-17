package tokenize

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
splitTokenizer encodes one Unicode character per token but reports
ErrInvalidUTF8 if asked to decode only the first half of a 2-byte
character. Used to exercise the buffer-until-valid behavior.
*/
type splitTokenizer struct{}

func (splitTokenizer) Encode(text string) ([]int, error) {
	out := make([]int, 0, len(text))

	for _, runeValue := range text {
		out = append(out, int(runeValue))
	}

	return out, nil
}

/*
Decode mimics byte-level BPE: token 200 alone is the first byte of a
2-byte UTF-8 sequence (invalid on its own); token 201 is the second
byte and decodes to "é". All other tokens decode to their rune.
*/
func (splitTokenizer) Decode(ids []int, skip bool) (string, error) {
	if len(ids) == 1 && ids[0] == 200 {
		return "", tokenizer.ErrInvalidUTF8
	}

	if len(ids) == 2 && ids[0] == 200 && ids[1] == 201 {
		return "é", nil
	}

	out := make([]byte, 0, len(ids))

	for _, id := range ids {
		out = append(out, byte(id))
	}

	return string(out), nil
}

func (splitTokenizer) VocabSize() int         { return 1024 }
func (splitTokenizer) SpecialTokenIDs() []int { return nil }

func TestStreamDecode(t *testing.T) {
	Convey("Given a tokenizer that splits UTF-8 across two tokens", t, func() {
		stub := optest.NewStubContext()
		stub.Tokenizers["tok"] = splitTokenizer{}

		stream, err := state.Default.Build("token_stream", "stream", nil)
		So(err, ShouldBeNil)
		stub.States["stream"] = stream

		Convey("First half-token should buffer and emit empty text", func() {
			stub.Scope["tok_in"] = 200
			stub.StepRef = program.Step{
				ID: "stream",
				Op: "tokenizer.stream_decode",
				Inputs: map[string]program.ValueRef{
					"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
					"stream":    {Namespace: program.NamespaceState, Name: "stream"},
					"token":     {Namespace: program.NamespaceLocal, Name: "tok_in"},
				},
				Outputs: map[string]program.ValueRef{
					"text": {Namespace: program.NamespaceLocal, Name: "decoded"},
				},
			}

			So(StreamDecode{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["decoded"], ShouldEqual, "")
			So(stream.(*state.TokenStream).Pending(), ShouldResemble, []int{200})
		})

		Convey("Second half-token should flush the rune and clear the buffer", func() {
			stub.Scope["tok_in"] = 200
			stub.StepRef = program.Step{
				ID: "decode_first",
				Op: "tokenizer.stream_decode",
				Inputs: map[string]program.ValueRef{
					"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
					"stream":    {Namespace: program.NamespaceState, Name: "stream"},
					"token":     {Namespace: program.NamespaceLocal, Name: "tok_in"},
				},
				Outputs: map[string]program.ValueRef{
					"text": {Namespace: program.NamespaceLocal, Name: "decoded"},
				},
			}
			So(StreamDecode{}.Execute(stub), ShouldBeNil)

			stub.Scope["tok_in"] = 201
			So(StreamDecode{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["decoded"], ShouldEqual, "é")
			So(stream.(*state.TokenStream).Pending(), ShouldBeEmpty)
		})
	})
}

func TestStreamFlush(t *testing.T) {
	Convey("Given a stream with no buffered tokens", t, func() {
		stub := optest.NewStubContext()
		stub.Tokenizers["tok"] = splitTokenizer{}

		stream, err := state.Default.Build("token_stream", "stream", nil)
		So(err, ShouldBeNil)
		stub.States["stream"] = stream

		stub.StepRef = program.Step{
			ID: "flush",
			Op: "tokenizer.stream_flush",
			Inputs: map[string]program.ValueRef{
				"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
				"stream":    {Namespace: program.NamespaceState, Name: "stream"},
			},
			Outputs: map[string]program.ValueRef{
				"text": {Namespace: program.NamespaceLocal, Name: "remaining"},
			},
		}

		Convey("Flush should emit empty string without error", func() {
			So(StreamFlush{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["remaining"], ShouldEqual, "")
		})
	})

	Convey("Given a stream holding an incomplete UTF-8 sequence", t, func() {
		stub := optest.NewStubContext()
		stub.Tokenizers["tok"] = splitTokenizer{}

		stream, err := state.Default.Build("token_stream", "stream", nil)
		So(err, ShouldBeNil)
		stream.(*state.TokenStream).Append(200)
		stub.States["stream"] = stream

		stub.StepRef = program.Step{
			ID: "flush",
			Op: "tokenizer.stream_flush",
			Inputs: map[string]program.ValueRef{
				"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
				"stream":    {Namespace: program.NamespaceState, Name: "stream"},
			},
			Outputs: map[string]program.ValueRef{
				"text": {Namespace: program.NamespaceLocal, Name: "remaining"},
			},
		}

		Convey("Flush should surface the invalid-UTF-8 error", func() {
			err := StreamFlush{}.Execute(stub)
			So(err, ShouldNotBeNil)
		})
	})
}
