package program

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestProgramValidate(t *testing.T) {
	Convey("Given a runtime Program", t, func() {
		valid := func() *Program {
			return &Program{
				Name: "chat",
				Assets: []AssetDeclaration{
					{ID: "model", Kind: "model", Source: "openai-community/gpt2"},
					{ID: "tokenizer", Kind: "tokenizer", Source: "openai-community/gpt2"},
				},
				State: []StateDeclaration{
					{ID: "history", Type: "token_buffer"},
					{ID: "position", Type: "counter"},
				},
				Samplers: []SamplerDeclaration{
					{ID: "main", Type: "categorical"},
				},
				Graphs: map[string]GraphModule{
					"forward": {ID: "forward", Topology: "system.topology"},
				},
				Steps: []Step{
					{
						ID: "read_user",
						Op: "io.read_line",
						Outputs: map[string]ValueRef{
							"text": {Namespace: NamespaceLocal, Name: "user_text"},
						},
					},
					{
						ID: "generate",
						Op: "control.loop_count",
						Config: map[string]any{
							"count": 8,
						},
						Body: []Step{
							{
								ID: "forward",
								Op: "graph.call",
								Inputs: map[string]ValueRef{
									"graph": {Namespace: NamespaceGraph, Name: "forward"},
									"kv":    {Namespace: NamespaceState, Name: "history"},
								},
								Outputs: map[string]ValueRef{
									"logits": {Namespace: NamespaceLocal, Name: "logits"},
								},
							},
							{
								ID: "sample",
								Op: "sampler.next_token",
								Inputs: map[string]ValueRef{
									"sampler": {Namespace: NamespaceSampler, Name: "main"},
									"logits":  {Namespace: NamespaceLocal, Name: "logits"},
								},
								Outputs: map[string]ValueRef{
									"token": {Namespace: NamespaceLocal, Name: "next_token"},
								},
							},
						},
					},
				},
			}
		}

		Convey("It should accept a well-formed program", func() {
			So(valid().Validate(), ShouldBeNil)
		})

		Convey("It should reject a missing name", func() {
			program := valid()
			program.Name = ""
			So(program.Validate(), ShouldNotBeNil)
		})

		Convey("It should reject a step referencing an undeclared state", func() {
			program := valid()
			program.Steps[1].Body[0].Inputs["kv"] = ValueRef{
				Namespace: NamespaceState,
				Name:      "missing",
			}
			So(program.Validate(), ShouldNotBeNil)
		})

		Convey("It should reject a step referencing an undeclared graph", func() {
			program := valid()
			program.Steps[1].Body[0].Inputs["graph"] = ValueRef{
				Namespace: NamespaceGraph,
				Name:      "missing",
			}
			So(program.Validate(), ShouldNotBeNil)
		})

		Convey("It should reject a step referencing an undeclared sampler", func() {
			program := valid()
			program.Steps[1].Body[1].Inputs["sampler"] = ValueRef{
				Namespace: NamespaceSampler,
				Name:      "missing",
			}
			So(program.Validate(), ShouldNotBeNil)
		})

		Convey("It should reject duplicate step ids inside a body", func() {
			program := valid()
			program.Steps[1].Body = append(program.Steps[1].Body, Step{
				ID: "forward",
				Op: "value.assign",
			})
			So(program.Validate(), ShouldNotBeNil)
		})

		Convey("FindStep should descend into nested bodies", func() {
			program := valid()
			step := program.FindStep("sample")
			So(step, ShouldNotBeNil)
			So(step.Op, ShouldEqual, OperationID("sampler.next_token"))
		})

		Convey("StateByID and AssetByID should return the declaration", func() {
			program := valid()
			So(program.StateByID("history"), ShouldNotBeNil)
			So(program.AssetByID("tokenizer"), ShouldNotBeNil)
			So(program.SamplerByID("main"), ShouldNotBeNil)
			So(program.StateByID("missing"), ShouldBeNil)
		})
	})
}
