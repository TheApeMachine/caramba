package manifest

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/asset"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestCompiler_CompileFlux2Klein4B(test *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B diffusion manifest", test, func() {
		// The runtime manifest pulls its topology from an architecture
		// include, so the compiler must read through the asset FS so
		// the include can resolve against sibling embedded files. The
		// old CompileBytes path could not see siblings and would fail
		// on the include directive.
		_, err := asset.ReadFile("model/diffusion/flux-2-klein-4b.yml")
		So(err, ShouldBeNil)

		compiler := NewCompiler(".").WithFS(asset.TemplateFS())

		Convey("It should compile the denoiser topology", func() {
			graph, err := compiler.Compile("model/diffusion/flux-2-klein-4b.yml")

			So(err, ShouldBeNil)
			So(graph, ShouldNotBeNil)
			So(graph.ExternalInputs(), ShouldResemble, []string{
				"encoder_hidden_states",
				"hidden_states",
				"timestep",
			})
			// denoiser_output is the final node now — it slices the
			// joint latent+text sequence back down to just the latent
			// 4096 tokens before the runtime hands the result to the
			// flow-match scheduler.
			So(graph.nodes[len(graph.nodes)-1].ID, ShouldEqual, "denoiser_output")
		})

		Convey("It should lower to IR with latent input and output shapes", func() {
			graph, err := compiler.Compile("model/diffusion/flux-2-klein-4b.yml")
			So(err, ShouldBeNil)

			// 4096 latent tokens matches a real 1024x1024 generation
			// (1024*1024 / 16^2); the joint dual-stream concat with the
			// 16-token prompt gives a 4112-position joint sequence
			// which the final shape.slice trims back to 4096.
			latentShape, err := tensor.NewShape([]int{1, 4096, 128})
			So(err, ShouldBeNil)
			promptShape, err := tensor.NewShape([]int{1, 16, 7680})
			So(err, ShouldBeNil)
			timestepShape, err := tensor.NewShape([]int{1})
			So(err, ShouldBeNil)

			irGraph, err := LowerGraphToIRWithInputShapes(
				graph,
				latentShape,
				map[string]tensor.Shape{
					"encoder_hidden_states": promptShape,
					"timestep":              timestepShape,
				},
			)
			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			// proj_out emits the joint sequence projected to 128 ch.
			projOutNode := index.Node("proj_out")
			So(projOutNode, ShouldNotBeNil)
			So(projOutNode.Shape().Dims(), ShouldResemble, []int{1, 4112, 128})

			// denoiser_output is proj_out sliced back to just latents.
			denoiserOutput := index.Node("denoiser_output")
			So(denoiserOutput, ShouldNotBeNil)
			So(denoiserOutput.Shape().Dims(), ShouldResemble, []int{1, 4096, 128})

			// The to_q projection sits inside the first dual transformer
			// block, which operates on the *joint* latent+text stream
			// after conditioning_concat — so its input/output sequence
			// length is 4096 + 16 = 4112, not 4096. The slice back down
			// to 4096 only happens at the very end (denoiser_output).
			queryNode := index.Node("transformer_blocks.0.attn.to_q")
			So(queryNode, ShouldNotBeNil)
			So(queryNode.Shape().Dims(), ShouldResemble, []int{1, 4112, 3072})

			contextNode := index.Node("context_embedder")
			So(contextNode, ShouldNotBeNil)
			So(contextNode.Shape().Dims(), ShouldResemble, []int{1, 16, 3072})
		})
	})
}

func TestCompiler_CompileFlux2Klein4BTextEncoder(test *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B text encoder manifest", test, func() {
		data, err := asset.ReadFile("model/diffusion/flux-2-klein-4b-text-encoder.yml")
		So(err, ShouldBeNil)

		compiler := NewCompiler(".")

		Convey("It should compile the Qwen3 prompt encoder topology", func() {
			graph, err := compiler.CompileBytes(data)

			So(err, ShouldBeNil)
			So(graph, ShouldNotBeNil)
			So(graph.ExternalInputs(), ShouldResemble, []string{"input_ids"})
			So(graph.nodes[len(graph.nodes)-1].ID, ShouldEqual, "prompt_embeds")
		})

		Convey("It should lower hidden-state concatenation to 7680-wide prompt embeddings", func() {
			graph, err := compiler.CompileBytes(data)
			So(err, ShouldBeNil)

			tokenShape, err := tensor.NewShape([]int{1, 8})
			So(err, ShouldBeNil)

			irGraph, err := LowerGraphToIR(graph, tokenShape)
			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			outputNode := index.Node("prompt_embeds")
			So(outputNode, ShouldNotBeNil)
			So(outputNode.Shape().Dims(), ShouldResemble, []int{1, 8, 7680})

			queryNode := index.Node("model.layers.0.self_attn.q_proj")
			So(queryNode, ShouldNotBeNil)
			So(queryNode.Shape().Dims(), ShouldResemble, []int{1, 8, 4096})
		})
	})
}

func TestCompiler_CompileFlux2Klein4BVAE(test *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B VAE decoder manifest", test, func() {
		data, err := asset.ReadFile("model/diffusion/flux-2-klein-4b-vae-decoder.yml")
		So(err, ShouldBeNil)

		compiler := NewCompiler(".")

		Convey("It should unpack packed latent tokens into channel-first VAE latents", func() {
			graph, err := compiler.CompileBytes(data)
			So(err, ShouldBeNil)

			gridNode := graph.index["vae.unpack.grid"]
			So(gridNode, ShouldNotBeNil)
			So(gridNode.Config["shape"], ShouldResemble, []any{1, 64, 64, 32, 2, 2})

			t23Node := graph.index["vae.unpack.t23"]
			So(t23Node, ShouldNotBeNil)
			So(t23Node.In, ShouldResemble, []string{"unpack_grid"})
			So(t23Node.Out, ShouldResemble, []string{"unpack_t23"})
			So(t23Node.Config["dim0"], ShouldEqual, 2)
			So(t23Node.Config["dim1"], ShouldEqual, 3)

			t12Node := graph.index["vae.unpack.t12"]
			So(t12Node, ShouldNotBeNil)
			So(t12Node.In, ShouldResemble, []string{"unpack_t23"})
			So(t12Node.Out, ShouldResemble, []string{"unpack_t12"})
			So(t12Node.Config["dim0"], ShouldEqual, 1)
			So(t12Node.Config["dim1"], ShouldEqual, 2)

			t34Node := graph.index["vae.unpack.t34"]
			So(t34Node, ShouldNotBeNil)
			So(t34Node.In, ShouldResemble, []string{"unpack_t12"})
			So(t34Node.Out, ShouldResemble, []string{"unpack_t34"})
			So(t34Node.Config["dim0"], ShouldEqual, 3)
			So(t34Node.Config["dim1"], ShouldEqual, 4)

			latentNode := graph.index["vae.unpack.latent"]
			So(latentNode, ShouldNotBeNil)
			So(latentNode.In, ShouldResemble, []string{"unpack_t34"})
			So(latentNode.Config["shape"], ShouldResemble, []any{1, 32, 128, 128})
		})
	})
}

func BenchmarkCompiler_CompileFlux2Klein4B(benchmark *testing.B) {
	if _, err := asset.ReadFile("model/diffusion/flux-2-klein-4b.yml"); err != nil {
		benchmark.Fatal(err)
	}

	compiler := NewCompiler(".").WithFS(asset.TemplateFS())

	for benchmark.Loop() {
		if _, err := compiler.Compile("model/diffusion/flux-2-klein-4b.yml"); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkCompiler_CompileFlux2Klein4BVAE(benchmark *testing.B) {
	data, err := asset.ReadFile("model/diffusion/flux-2-klein-4b-vae-decoder.yml")

	if err != nil {
		benchmark.Fatal(err)
	}

	compiler := NewCompiler(".")

	for benchmark.Loop() {
		if _, err := compiler.CompileBytes(data); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkCompiler_CompileFlux2Klein4BTextEncoder(benchmark *testing.B) {
	data, err := asset.ReadFile("model/diffusion/flux-2-klein-4b-text-encoder.yml")

	if err != nil {
		benchmark.Fatal(err)
	}

	compiler := NewCompiler(".")

	for benchmark.Loop() {
		if _, err := compiler.CompileBytes(data); err != nil {
			benchmark.Fatal(err)
		}
	}
}
