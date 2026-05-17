package chat

import (
	"context"
	"fmt"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

const deterministicChatYAML = `
name: chat_test
system:
  runtime:
    type: program
    entry: chat
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

type tinyModelRunner struct {
	vocabSize int
}

func (tiny tinyModelRunner) Call(
	execContext context.Context,
	module program.GraphModule,
	inputs map[string]any,
) (map[string]any, error) {
	history, ok := inputs["history"].(*state.TokenBuffer)

	if !ok {
		return nil, fmt.Errorf("tinyModelRunner: missing history token_buffer")
	}

	if _, ok := inputs["input_ids"]; !ok {
		return nil, fmt.Errorf("tinyModelRunner: missing input_ids")
	}

	logits := make([]float64, tiny.vocabSize)
	target := history.Length() % tiny.vocabSize

	for index := range logits {
		logits[index] = -10.0
	}

	logits[target] = 10.0

	return map[string]any{"logits": logits}, nil
}

type tinyTokenizer struct{}

func (tinyTokenizer) Encode(text string) ([]int, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}

	out := []int{}

	for _, field := range strings.Fields(text) {
		var value int

		if _, err := fmt.Sscanf(field, "%d", &value); err != nil {
			value = len(field) % 5
		}

		out = append(out, value)
	}

	return out, nil
}

func (tinyTokenizer) Decode(ids []int, skip bool) (string, error) {
	pieces := make([]string, len(ids))

	for index, id := range ids {
		pieces[index] = fmt.Sprintf("<%d>", id)
	}

	return strings.Join(pieces, ""), nil
}

func (tinyTokenizer) VocabSize() int         { return 5 }
func (tinyTokenizer) SpecialTokenIDs() []int { return []int{4} }

func TestRuntimeGeneratorGenerate(t *testing.T) {
	Convey("Given a RuntimeGenerator wired to a deterministic tiny model", t, func() {
		runtimeProgram, err := compiler.New(".").CompileBytes([]byte(deterministicChatYAML))
		So(err, ShouldBeNil)

		generator, err := NewRuntimeGenerator(RuntimeGeneratorOptions{
			Program:     runtimeProgram,
			Tokenizer:   tinyTokenizer{},
			GraphRunner: tinyModelRunner{vocabSize: 5},
		})
		So(err, ShouldBeNil)

		Convey("Generate should emit the predicted tokens through the emit callback", func() {
			buffer := strings.Builder{}
			err := generator.Generate(context.Background(), "", func(chunk string) error {
				buffer.WriteString(chunk)

				return nil
			})

			So(err, ShouldBeNil)
			So(buffer.String(), ShouldEqual, "<0><1><2><3><4>")
		})

		Convey("Generate should surface emit-callback errors", func() {
			err := generator.Generate(context.Background(), "", func(chunk string) error {
				return fmt.Errorf("downstream gone")
			})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "downstream gone")
		})

		Convey("Generate without an emit callback should error", func() {
			err := generator.Generate(context.Background(), "", nil)
			So(err, ShouldNotBeNil)
		})
	})

	Convey("NewRuntimeGenerator should require a tokenizer and a graph runner", t, func() {
		_, err := NewRuntimeGenerator(RuntimeGeneratorOptions{})
		So(err, ShouldNotBeNil)

		_, err = NewRuntimeGenerator(RuntimeGeneratorOptions{
			Tokenizer: tinyTokenizer{},
		})
		So(err, ShouldNotBeNil)
		So(err.Error(), ShouldContainSubstring, "graph runner")
	})
}
