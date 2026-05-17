package compiler

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

const minimalChatYAML = `
name: chat
system:
  runtime:
    type: program
    entry: chat
    backend: metal
    assets:
      model: { source: tiny/model }
      tokenizer: { source: tiny/tokenizer }
    state:
      history: { type: token_buffer }
      position: { type: counter, initial: 0 }
    samplers:
      main: { type: categorical, temperature: 0.8, top_k: 50 }
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
        count: 4
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

func TestCompileChatProgram(t *testing.T) {
	Convey("Given the minimal chat manifest", t, func() {
		runtimeProgram, err := New(".").CompileBytes([]byte(minimalChatYAML))
		So(err, ShouldBeNil)
		So(runtimeProgram, ShouldNotBeNil)

		Convey("Top-level metadata should be parsed", func() {
			So(runtimeProgram.Name, ShouldEqual, "chat")
			So(runtimeProgram.Entry, ShouldEqual, "chat")
			So(runtimeProgram.Backend, ShouldEqual, "metal")
		})

		Convey("Assets should be sorted by id and inferred kinds applied", func() {
			So(len(runtimeProgram.Assets), ShouldEqual, 2)
			So(runtimeProgram.Assets[0].ID, ShouldEqual, "model")
			So(runtimeProgram.Assets[0].Kind, ShouldEqual, "model")
			So(runtimeProgram.Assets[1].ID, ShouldEqual, "tokenizer")
			So(runtimeProgram.Assets[1].Kind, ShouldEqual, "tokenizer")
		})

		Convey("State declarations should preserve type and config", func() {
			So(len(runtimeProgram.State), ShouldEqual, 2)
			So(runtimeProgram.StateByID("history"), ShouldNotBeNil)
			So(runtimeProgram.StateByID("history").Type, ShouldEqual, "token_buffer")

			position := runtimeProgram.StateByID("position")
			So(position, ShouldNotBeNil)
			So(position.Type, ShouldEqual, "counter")
			So(position.Config["initial"], ShouldEqual, 0)
		})

		Convey("Sampler should be sorted by id and config preserved", func() {
			So(len(runtimeProgram.Samplers), ShouldEqual, 1)
			So(runtimeProgram.Samplers[0].ID, ShouldEqual, "main")
			So(runtimeProgram.Samplers[0].Config["temperature"], ShouldEqual, 0.8)
			So(runtimeProgram.Samplers[0].Config["top_k"], ShouldEqual, 50)
		})

		Convey("Graphs should be normalized into the program map", func() {
			module, ok := runtimeProgram.Graphs["forward"]
			So(ok, ShouldBeTrue)
			So(module.Topology, ShouldEqual, "tiny.topology")
		})

		Convey("Top-level steps should compile with normalized shortcuts", func() {
			So(len(runtimeProgram.Steps), ShouldEqual, 3)

			encodeStep := runtimeProgram.Steps[1]
			So(encodeStep.Op, ShouldEqual, program.OperationID("tokenizer.encode"))
			So(encodeStep.Inputs["tokenizer"].Namespace, ShouldEqual, program.NamespaceTokenizer)
			So(encodeStep.Inputs["tokenizer"].Name, ShouldEqual, "tokenizer")
			So(encodeStep.Inputs["text"].Namespace, ShouldEqual, program.NamespaceLocal)
			So(encodeStep.Inputs["text"].Name, ShouldEqual, "user_text")
			So(encodeStep.Outputs["tokens"].Name, ShouldEqual, "input_ids")
		})

		Convey("Nested body steps should compile recursively", func() {
			generate := runtimeProgram.Steps[2]
			So(generate.Op, ShouldEqual, program.OperationID("control.loop_count"))
			So(generate.Config["count"], ShouldEqual, 4)
			So(len(generate.Body), ShouldEqual, 6)

			forward := generate.Body[0]
			So(forward.Inputs["graph"].Namespace, ShouldEqual, program.NamespaceGraph)
			So(forward.Inputs["graph"].Name, ShouldEqual, "forward")
			So(forward.Inputs["input_ids"].Namespace, ShouldEqual, program.NamespaceLocal)
			So(forward.Inputs["history"].Namespace, ShouldEqual, program.NamespaceState)
			So(forward.Inputs["history"].Name, ShouldEqual, "history")

			sample := generate.Body[1]
			So(sample.Inputs["sampler"].Namespace, ShouldEqual, program.NamespaceSampler)
			So(sample.Outputs["stopped"].Name, ShouldEqual, "stopped")
		})
	})
}

func TestCompileMissingRuntime(t *testing.T) {
	Convey("A document without system.runtime should error clearly", t, func() {
		_, err := New(".").CompileBytes([]byte("name: nothing\n"))
		So(err, ShouldNotBeNil)
		So(err.Error(), ShouldContainSubstring, "system.runtime")
	})
}

func TestCompileRejectsWrongType(t *testing.T) {
	Convey("A document whose runtime.type is not 'program' should error", t, func() {
		yaml := "system:\n  runtime:\n    type: experiment\n    program: []\n"
		_, err := New(".").CompileBytes([]byte(yaml))
		So(err, ShouldNotBeNil)
		So(err.Error(), ShouldContainSubstring, "type must be 'program'")
	})
}

func TestCompileRejectsUndeclaredReference(t *testing.T) {
	Convey("Referencing an undeclared state object should fail validation", t, func() {
		yaml := `
name: broken
system:
  runtime:
    type: program
    state:
      history: { type: token_buffer }
    program:
      - id: append
        op: state.append
        value: token
        outputs: { target: state.missing }
`
		_, err := New(".").CompileBytes([]byte(yaml))
		So(err, ShouldNotBeNil)
		So(err.Error(), ShouldContainSubstring, "undeclared state")
	})
}
