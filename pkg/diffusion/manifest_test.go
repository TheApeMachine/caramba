package diffusion

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
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
			So(config.VAE.File, ShouldEqual, "vae/diffusion_pytorch_model.safetensors")
			So(config.Generation.Output, ShouldEqual, "flux-2-klein-4b.png")
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
