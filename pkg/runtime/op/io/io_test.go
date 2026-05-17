package io

import (
	"bytes"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

type fakeTokenizer struct{}

func (fakeTokenizer) Encode(text string) ([]int, error)                       { return nil, nil }
func (fakeTokenizer) Decode(tokenIDs []int, skip bool) (string, error)        { return "<<token>>", nil }
func (fakeTokenizer) VocabSize() int                                          { return 0 }
func (fakeTokenizer) SpecialTokenIDs() []int                                  { return nil }

func TestReadLine(t *testing.T) {
	Convey("Given a stdin with a single line", t, func() {
		stub := optest.NewStubContext()
		stub.StdinBuf = bytes.NewBufferString("hello world\n")
		stub.StepRef = program.Step{
			ID: "read",
			Op: "io.read_line",
			Outputs: map[string]program.ValueRef{
				"text": {Namespace: program.NamespaceLocal, Name: "text"},
			},
		}

		Convey("Execute should bind the trimmed line", func() {
			So(newReadLine().Execute(stub), ShouldBeNil)
			So(stub.Scope["text"], ShouldEqual, "hello world")
		})
	})
}

func TestEmitText(t *testing.T) {
	Convey("Given an EmitText op with text bound to a local", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["msg"] = "ready\n"
		stub.StepRef = program.Step{
			ID: "say",
			Op: "io.emit_text",
			Inputs: map[string]program.ValueRef{
				"text": {Namespace: program.NamespaceLocal, Name: "msg"},
			},
		}

		Convey("Execute should write the text to stdout", func() {
			So(EmitText{}.Execute(stub), ShouldBeNil)
			So(stub.StdoutBuf.String(), ShouldEqual, "ready\n")
		})
	})
}

func TestEmitToken(t *testing.T) {
	Convey("Given an EmitToken op with a tokenizer wired up", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["next_token"] = 7
		stub.Tokenizers["tok"] = fakeTokenizer{}
		stub.StepRef = program.Step{
			ID: "emit",
			Op: "io.emit_token",
			Inputs: map[string]program.ValueRef{
				"token":     {Namespace: program.NamespaceLocal, Name: "next_token"},
				"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tok"},
			},
		}

		Convey("Execute should write the decoded text to stdout", func() {
			So(EmitToken{}.Execute(stub), ShouldBeNil)
			So(stub.StdoutBuf.String(), ShouldEqual, "<<token>>")
		})
	})
}
