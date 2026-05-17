package integration

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/executor"
	"github.com/theapemachine/caramba/pkg/runtime/op"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/sampler"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
tinyModelRunner is a deterministic stand-in for a real transformer
forward graph. It produces logits that always favor the token whose
id equals (history_length % vocabSize). This is enough to drive the
runtime program from end to end without depending on the backend.
*/
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

/*
tinyTokenizer formats every token id as "<n>" so the emitted stdout
is easy to inspect in tests.
*/
type tinyTokenizer struct{}

func (tinyTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	tokens := []int{}

	for _, field := range strings.Fields(text) {
		var value int

		_, err := fmt.Sscanf(field, "%d", &value)

		if err != nil {
			return nil, err
		}

		tokens = append(tokens, value)
	}

	return tokens, nil
}

func (tinyTokenizer) Decode(ids []int, skip bool) (string, error) {
	parts := make([]string, len(ids))

	for index, id := range ids {
		parts[index] = fmt.Sprintf("<%d>", id)
	}

	return strings.Join(parts, ""), nil
}

func (tinyTokenizer) VocabSize() int         { return 5 }
func (tinyTokenizer) SpecialTokenIDs() []int { return []int{4} }

func chatProgram(maxNewTokens int) *program.Program {
	return &program.Program{
		Name:  "chat",
		Entry: "chat",
		Assets: []program.AssetDeclaration{
			{ID: "model", Kind: "model", Source: "tiny/model"},
			{ID: "tokenizer", Kind: "tokenizer", Source: "tiny/tokenizer"},
		},
		State: []program.StateDeclaration{
			{ID: "history", Type: "token_buffer"},
			{ID: "position", Type: "counter", Config: map[string]any{"initial": 0}},
		},
		Samplers: []program.SamplerDeclaration{
			{
				ID:   "main",
				Type: "categorical",
				Config: map[string]any{
					"temperature":    0.001,
					"top_k":          1,
					"stop_token_ids": []int{4},
				},
			},
		},
		Graphs: map[string]program.GraphModule{
			"forward": {ID: "forward", Topology: "tiny.topology"},
		},
		Steps: []program.Step{
			{
				ID: "read_user",
				Op: "io.read_line",
				Outputs: map[string]program.ValueRef{
					"text": {Namespace: program.NamespaceLocal, Name: "user_text"},
				},
			},
			{
				ID: "encode_user",
				Op: "tokenizer.encode",
				Inputs: map[string]program.ValueRef{
					"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tokenizer"},
					"text":      {Namespace: program.NamespaceLocal, Name: "user_text"},
				},
				Outputs: map[string]program.ValueRef{
					"tokens": {Namespace: program.NamespaceLocal, Name: "input_ids"},
				},
			},
			{
				ID:     "generate",
				Op:     "control.loop_count",
				Config: map[string]any{"count": maxNewTokens},
				Body: []program.Step{
					{
						ID: "forward",
						Op: "graph.call",
						Inputs: map[string]program.ValueRef{
							"graph":     {Namespace: program.NamespaceGraph, Name: "forward"},
							"input_ids": {Namespace: program.NamespaceLocal, Name: "input_ids"},
							"history":   {Namespace: program.NamespaceState, Name: "history"},
						},
						Outputs: map[string]program.ValueRef{
							"logits": {Namespace: program.NamespaceLocal, Name: "logits"},
						},
					},
					{
						ID: "sample",
						Op: "sampler.next_token",
						Inputs: map[string]program.ValueRef{
							"sampler": {Namespace: program.NamespaceSampler, Name: "main"},
							"logits":  {Namespace: program.NamespaceLocal, Name: "logits"},
							"history": {Namespace: program.NamespaceState, Name: "history"},
						},
						Outputs: map[string]program.ValueRef{
							"token":   {Namespace: program.NamespaceLocal, Name: "next_token"},
							"stopped": {Namespace: program.NamespaceLocal, Name: "stopped"},
						},
					},
					{
						ID: "append_history",
						Op: "state.append",
						Inputs: map[string]program.ValueRef{
							"value": {Namespace: program.NamespaceLocal, Name: "next_token"},
						},
						Outputs: map[string]program.ValueRef{
							"target": {Namespace: program.NamespaceState, Name: "history"},
						},
					},
					{
						ID: "emit",
						Op: "io.emit_token",
						Inputs: map[string]program.ValueRef{
							"tokenizer": {Namespace: program.NamespaceTokenizer, Name: "tokenizer"},
							"token":     {Namespace: program.NamespaceLocal, Name: "next_token"},
						},
					},
					{
						ID: "stop",
						Op: "control.break_if",
						Inputs: map[string]program.ValueRef{
							"condition": {Namespace: program.NamespaceLocal, Name: "stopped"},
						},
					},
					{
						ID: "carry_token",
						Op: "value.assign",
						Inputs: map[string]program.ValueRef{
							"value": {Namespace: program.NamespaceLocal, Name: "next_token"},
						},
						Outputs: map[string]program.ValueRef{
							"target": {Namespace: program.NamespaceLocal, Name: "input_ids"},
						},
					},
					{
						ID:     "advance_position",
						Op:     "state.update",
						Config: map[string]any{"update": "increment"},
						Outputs: map[string]program.ValueRef{
							"target": {Namespace: program.NamespaceState, Name: "position"},
						},
					},
				},
			},
		},
	}
}

func TestChatRuntimeEndToEnd(t *testing.T) {
	Convey("Given the canonical chat runtime program wired to a tiny model", t, func() {
		stdin := bytes.NewBufferString("\n")
		stdout := &bytes.Buffer{}

		runner := tinyModelRunner{vocabSize: 5}
		categoricalSampler := sampler.New(12345)

		runtimeProgram := chatProgram(16)

		exec, err := executor.New(executor.Options{
			Program:       runtimeProgram,
			Tokenizers:    map[string]tokenizer.Tokenizer{"tokenizer": tinyTokenizer{}},
			GraphRunner:   runner,
			SamplerRunner: categoricalSampler,
			Stdin:         stdin,
			Stdout:        stdout,
		})
		So(err, ShouldBeNil)

		Convey("Run should drive the loop to the stop token and emit each generated token", func() {
			So(exec.Run(context.Background()), ShouldBeNil)

			Convey("Stdout should contain the 5 generated tokens", func() {
				So(stdout.String(), ShouldEqual, "<0><1><2><3><4>")
			})

			Convey("History state should hold the same five tokens", func() {
				historyState, err := exec.States()["history"].(*state.TokenBuffer), error(nil)
				So(err, ShouldBeNil)
				So(historyState.Tokens(), ShouldResemble, []int{0, 1, 2, 3, 4})
			})

			Convey("Position counter should reflect 4 successful carry+advance pairs", func() {
				positionState := exec.States()["position"].(*state.Counter)
				So(positionState.Value(), ShouldEqual, 4)
			})
		})
	})
}

// silence unused-import vet warnings if op surface changes in the future.
var _ = op.Default
