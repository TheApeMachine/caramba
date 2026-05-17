package runtime

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

const deterministicModelRuntimeYAML = `
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
          - id: make_position_ids
            op: value.positions
            inputs:
              start: state.position
              tokens: input_ids
            outputs:
              positions: position_ids
          - id: measure_input
            op: value.length
            inputs:
              value: input_ids
            outputs:
              length: input_token_count
          - id: forward
            op: graph.call
            graph: forward
            inputs:
              input_ids: input_ids
              position_ids: position_ids
              position_start: state.position
              history: state.history
            outputs:
              logits: logits
          - id: advance_position
            op: state.update
            update: increment
            inputs:
              delta: input_token_count
            target: state.position
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
            target: state.history
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
`

type tinyRuntimeModelRunner struct {
	vocabSize int
}

func (runner tinyRuntimeModelRunner) Call(
	execContext context.Context,
	module program.GraphModule,
	inputs map[string]any,
) (map[string]any, error) {
	history, ok := inputs["history"].(*state.TokenBuffer)

	if !ok {
		return nil, fmt.Errorf("tinyRuntimeModelRunner: missing history token_buffer")
	}

	if _, ok := inputs["input_ids"]; !ok {
		return nil, fmt.Errorf("tinyRuntimeModelRunner: missing input_ids")
	}

	if _, ok := inputs["position_ids"]; !ok {
		return nil, fmt.Errorf("tinyRuntimeModelRunner: missing position_ids")
	}

	if _, ok := inputs["position_start"]; !ok {
		return nil, fmt.Errorf("tinyRuntimeModelRunner: missing position_start")
	}

	logits := make([]float64, runner.vocabSize)
	target := history.Length() % runner.vocabSize

	for index := range logits {
		logits[index] = -10
	}

	logits[target] = 10

	return map[string]any{"logits": logits}, nil
}

type tinyRuntimeTokenizer struct{}

func (tinyRuntimeTokenizer) Encode(text string) ([]int, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}

	tokens := []int{}

	for _, field := range strings.Fields(text) {
		var value int

		if _, err := fmt.Sscanf(field, "%d", &value); err != nil {
			value = len(field) % 5
		}

		tokens = append(tokens, value)
	}

	return tokens, nil
}

func (tinyRuntimeTokenizer) Decode(ids []int, skip bool) (string, error) {
	pieces := make([]string, len(ids))

	for index, id := range ids {
		pieces[index] = fmt.Sprintf("<%d>", id)
	}

	return strings.Join(pieces, ""), nil
}

func (tinyRuntimeTokenizer) VocabSize() int {
	return 5
}

func (tinyRuntimeTokenizer) SpecialTokenIDs() []int {
	return []int{4}
}

func TestRuntimeGeneratorGenerate(t *testing.T) {
	Convey("Given a RuntimeGenerator wired to a deterministic tiny model", t, func() {
		generator := newTestRuntimeGenerator(t)

		Convey("Generate should emit predicted tokens through the callback", func() {
			buffer := strings.Builder{}
			err := generator.Generate(context.Background(), "", func(chunk string) error {
				buffer.WriteString(chunk)

				return nil
			})

			So(err, ShouldBeNil)
			So(buffer.String(), ShouldEqual, "<0><1><2><3><4>")
		})

		Convey("Generate should surface callback errors", func() {
			err := generator.Generate(context.Background(), "", func(chunk string) error {
				return fmt.Errorf("downstream closed")
			})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "downstream closed")
		})

		Convey("Generate should require an emit callback", func() {
			err := generator.Generate(context.Background(), "", nil)
			So(err, ShouldNotBeNil)
		})
	})

	Convey("NewRuntimeGenerator should require a tokenizer and a graph runner", t, func() {
		_, err := NewRuntimeGenerator(RuntimeGeneratorOptions{})
		So(err, ShouldNotBeNil)

		_, err = NewRuntimeGenerator(RuntimeGeneratorOptions{
			Tokenizer: tinyRuntimeTokenizer{},
		})
		So(err, ShouldNotBeNil)
		So(err.Error(), ShouldContainSubstring, "graph runner")
	})
}

func TestSessionRun(t *testing.T) {
	Convey("Given a Session whose generator owns runtime execution", t, func() {
		generator := newTestRuntimeGenerator(t)
		input := bytes.NewBufferString("hello\n")
		output := &bytes.Buffer{}
		session := NewSession(context.Background(), input, output, generator)

		Convey("Run should stream manifest output", func() {
			So(session.Run(), ShouldBeNil)
			So(output.String(), ShouldContainSubstring, "<0><1><2><3><4>")
		})
	})
}

func BenchmarkRuntimeGenerator_Generate(benchmark *testing.B) {
	generator := newTestRuntimeGenerator(benchmark)

	for benchmark.Loop() {
		err := generator.Generate(context.Background(), "", func(chunk string) error {
			return nil
		})

		if err != nil {
			benchmark.Fatal(err)
		}
	}
}

func newTestRuntimeGenerator(testingObject interface {
	Helper()
	Fatalf(string, ...any)
}) *RuntimeGenerator {
	testingObject.Helper()

	runtimeProgram, err := compiler.New(".").CompileBytes([]byte(deterministicModelRuntimeYAML))

	if err != nil {
		testingObject.Fatalf("compile runtime: %v", err)
	}

	generator, err := NewRuntimeGenerator(RuntimeGeneratorOptions{
		Program:     runtimeProgram,
		Tokenizer:   tinyRuntimeTokenizer{},
		GraphRunner: tinyRuntimeModelRunner{vocabSize: 5},
	})

	if err != nil {
		testingObject.Fatalf("new generator: %v", err)
	}

	return generator
}
