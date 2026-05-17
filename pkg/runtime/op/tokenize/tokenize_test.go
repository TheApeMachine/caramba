package tokenize

import (
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

type fixedTokenizer struct{}

func (fixedTokenizer) Encode(text string) ([]int, error) {
	tokens := []int{}

	for _, segment := range strings.Fields(text) {
		tokens = append(tokens, len(segment))
	}

	return tokens, nil
}

func (fixedTokenizer) Decode(tokenIDs []int, skip bool) (string, error) {
	pieces := make([]string, len(tokenIDs))

	for index, id := range tokenIDs {
		pieces[index] = strings.Repeat("x", id)
	}

	return strings.Join(pieces, " "), nil
}

func (fixedTokenizer) VocabSize() int         { return 1024 }
func (fixedTokenizer) SpecialTokenIDs() []int { return nil }

func TestEncode(t *testing.T) {
	Convey("Given an Encode step over a tokenizer asset", t, func() {
		stub := optest.NewStubContext()
		stub.Tokenizers["tok"] = fixedTokenizer{}
		stub.Scope["prompt"] = "two words"

		stub.StepRef = program.Step{
			ID: "encode",
			Op: "tokenizer.encode",
			Inputs: map[string]program.ValueRef{
				"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
				"text":      {Namespace: program.NamespaceLocal, Name: "prompt"},
			},
			Outputs: map[string]program.ValueRef{
				"tokens": {Namespace: program.NamespaceLocal, Name: "tokens"},
			},
		}

		Convey("Execute should bind the encoded []int", func() {
			So(Encode{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["tokens"], ShouldResemble, []int{3, 5})
		})
	})
}

func TestDecode(t *testing.T) {
	Convey("Given a Decode step over a tokenizer asset", t, func() {
		stub := optest.NewStubContext()
		stub.Tokenizers["tok"] = fixedTokenizer{}
		stub.Scope["tokens"] = []int{1, 2}

		stub.StepRef = program.Step{
			ID: "decode",
			Op: "tokenizer.decode",
			Inputs: map[string]program.ValueRef{
				"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
				"tokens":    {Namespace: program.NamespaceLocal, Name: "tokens"},
			},
			Outputs: map[string]program.ValueRef{
				"text": {Namespace: program.NamespaceLocal, Name: "text"},
			},
		}

		Convey("Execute should bind the decoded string", func() {
			So(Decode{}.Execute(stub), ShouldBeNil)
			So(stub.Scope["text"], ShouldEqual, "x xx")
		})
	})
}
