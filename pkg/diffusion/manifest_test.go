package diffusion

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/manifest"
)

func TestResolveManifestConfig(test *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B diffusion manifest", test, func() {
		config, _, manifestPath, err := ResolveManifestConfig(Config{
			Manifest: DefaultManifest,
		})

		Convey("It should resolve runtime assets and generation defaults", func() {
			So(err, ShouldBeNil)
			So(manifestPath, ShouldEqual, DefaultManifest)
			So(config.Runtime, ShouldEqual, "diffusion")
			So(config.Model.Source, ShouldEqual, "black-forest-labs/FLUX.2-klein-4B")
			So(config.TextEncoder.Manifest, ShouldEqual, "model/diffusion/flux-2-klein-4b-text-encoder.yml")
			So(config.TextEncoder.File, ShouldEqual, "text_encoder/model.safetensors.index.json")
			So(config.Transformer.File, ShouldEqual, "transformer/diffusion_pytorch_model.safetensors")
			So(config.VAE.Manifest, ShouldEqual, "model/diffusion/flux-2-klein-4b-vae-decoder.yml")
			So(config.VAE.File, ShouldEqual, "vae/diffusion_pytorch_model.safetensors")
			So(config.Generation.Output, ShouldEqual, "flux-2-klein-4b.png")
		})
	})
}

func TestCompileManifest(test *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B VAE decoder manifest", test, func() {
		graph, compiledManifest, err := CompileManifest("model/diffusion/flux-2-klein-4b-vae-decoder.yml")

		Convey("It should compile and lower to an RGB decoder target", func() {
			So(err, ShouldBeNil)
			So(compiledManifest, ShouldEqual, "model/diffusion/flux-2-klein-4b-vae-decoder.yml")

			shape, err := tensor.NewShape([]int{1, 4096, 128})
			So(err, ShouldBeNil)

			irGraph, err := manifest.LowerGraphToIR(graph, shape)
			So(err, ShouldBeNil)

			target := outputTarget(irGraph)
			So(target, ShouldNotBeNil)
			So(target.ID(), ShouldEqual, "decoder.conv_out")
			So(target.Shape().Dims(), ShouldResemble, []int{1, 3, 1024, 1024})
		})
	})
}

func BenchmarkResolveManifestConfig(benchmark *testing.B) {
	for benchmark.Loop() {
		if _, _, _, err := ResolveManifestConfig(Config{
			Manifest: DefaultManifest,
		}); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkCompileManifest(benchmark *testing.B) {
	for benchmark.Loop() {
		if _, _, err := CompileManifest("model/diffusion/flux-2-klein-4b-vae-decoder.yml"); err != nil {
			benchmark.Fatal(err)
		}
	}
}
