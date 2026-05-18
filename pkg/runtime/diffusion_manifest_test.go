package runtime

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/manifest"
)

func TestResolveDiffusionConfig(t *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B diffusion manifest", t, func() {
		config, _, manifestPath, err := ResolveDiffusionConfig(DiffusionConfig{
			Manifest: DefaultDiffusionManifest,
		})

		Convey("It should resolve runtime assets and generation defaults", func() {
			So(err, ShouldBeNil)
			So(manifestPath, ShouldEqual, DefaultDiffusionManifest)
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

func TestCompileDiffusionManifest(t *testing.T) {
	Convey("Given the embedded FLUX.2 Klein 4B VAE decoder manifest", t, func() {
		graph, compiledManifest, err := CompileDiffusionManifest(
			"model/diffusion/flux-2-klein-4b-vae-decoder.yml",
		)

		Convey("It should compile and lower to a [1, 3, 1024, 1024] output sink", func() {
			So(err, ShouldBeNil)
			So(compiledManifest, ShouldEqual, "model/diffusion/flux-2-klein-4b-vae-decoder.yml")

			shape, err := tensor.NewShape([]int{1, 4096, 128})
			So(err, ShouldBeNil)

			irGraph, err := manifest.LowerGraphToIR(graph, shape)
			So(err, ShouldBeNil)

			sinks := irGraph.Sinks()
			So(len(sinks), ShouldBeGreaterThan, 0)

			target := sinks[0]
			So(target.ID(), ShouldEqual, "decoder.conv_out")
			So(target.Shape().Dims(), ShouldResemble, []int{1, 3, 1024, 1024})
		})
	})
}

func TestDefaultDiffusionRuntimeImageLayout(t *testing.T) {
	Convey("Given the embedded diffusion runtime manifest", t, func() {
		runtimeProgram, _, err := loadRuntimeProgram(
			"",
			diffusionRuntimeAsset,
			diffusionManifestPrefix,
		)
		So(err, ShouldBeNil)

		writeStep := runtimeProgram.FindStep("write_image")
		So(writeStep, ShouldNotBeNil)

		Convey("It should write the VAE's NCHW output as channel-planar RGB", func() {
			So(writeStep.Config["layout"], ShouldEqual, "channel_planar")
			So(writeStep.Config["channels"], ShouldEqual, 3)
			So(writeStep.Config["range"], ShouldEqual, "neg_one_one")
		})
	})
}

func TestValidateRuntimeGraphInputs(t *testing.T) {
	Convey("Given the FLUX.2 denoiser manifest", t, func() {
		graph, _, err := CompileDiffusionManifest(DefaultDiffusionManifest)
		So(err, ShouldBeNil)

		Convey("It should reject the unused timestep input", func() {
			err := validateRuntimeGraphInputs("denoiser", graph)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, `input "timestep" has no consumers`)
		})
	})
}

func BenchmarkResolveDiffusionConfig(benchmark *testing.B) {
	for benchmark.Loop() {
		if _, _, _, err := ResolveDiffusionConfig(DiffusionConfig{
			Manifest: DefaultDiffusionManifest,
		}); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkCompileDiffusionManifest(benchmark *testing.B) {
	for benchmark.Loop() {
		_, _, err := CompileDiffusionManifest(
			"model/diffusion/flux-2-klein-4b-vae-decoder.yml",
		)

		if err != nil {
			benchmark.Fatal(err)
		}
	}
}
