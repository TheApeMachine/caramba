package integration

import (
	"bytes"
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/sampler"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

const yamlChatManifest = `
name: chat
system:
  runtime:
    type: program
    entry: chat
    assets:
      model: { source: tiny/model }
      tokenizer: { source: tiny/tokenizer }
    state:
      history: { type: token_buffer }
      position: { type: counter, initial: 0 }
    samplers:
      main:
        type: categorical
        temperature: 0.001
        top_k: 1
        stop_token_ids: [4]
    graphs:
      forward: { topology: tiny.topology }
    program:
      - id: read_user
        op: io.read_line
        outputs: { text: user_text }
      - id: encode_user
        op: tokenizer.encode
        tokenizer: tokenizer
        text: user_text
        outputs: { tokens: input_ids }
      - id: generate
        op: control.loop_count
        count: 16
        body:
          - id: forward
            op: graph.call
            graph: forward
            inputs:
              input_ids: input_ids
              history: state.history
            outputs:
              logits: logits
          - id: sample
            op: sampler.next_token
            sampler: main
            logits: logits
            history: state.history
            outputs:
              token: next_token
              stopped: stopped
          - id: append_history
            op: state.append
            value: next_token
            outputs: { target: state.history }
          - id: emit
            op: io.emit_token
            tokenizer: tokenizer
            token: next_token
          - id: stop
            op: control.break_if
            condition: stopped
          - id: carry_token
            op: value.assign
            value: next_token
            outputs: { target: input_ids }
          - id: advance_position
            op: state.update
            update: increment
            outputs: { target: state.position }
`

func TestYAMLChatProgramEndToEnd(t *testing.T) {
	Convey("Given a YAML chat manifest", t, func() {
		runtimeProgram, err := compiler.New(".").CompileBytes([]byte(yamlChatManifest))
		So(err, ShouldBeNil)

		stdin := bytes.NewBufferString("\n")
		stdout := &bytes.Buffer{}

		exec, err := executor.New(executor.Options{
			Program:       runtimeProgram,
			Tokenizers:    map[string]tokenizer.Tokenizer{"tokenizer": tinyTokenizer{}},
			GraphRunner:   tinyModelRunner{vocabSize: 5},
			SamplerRunner: sampler.New(12345),
			Stdin:         stdin,
			Stdout:        stdout,
		})
		So(err, ShouldBeNil)

		Convey("Compiled program should execute the same as the Go-built one", func() {
			So(exec.Run(context.Background()), ShouldBeNil)
			So(stdout.String(), ShouldEqual, "<0><1><2><3><4>")

			history := exec.States()["history"].(*state.TokenBuffer)
			So(history.Tokens(), ShouldResemble, []int{0, 1, 2, 3, 4})

			position := exec.States()["position"].(*state.Counter)
			So(position.Value(), ShouldEqual, 4)
		})
	})
}
