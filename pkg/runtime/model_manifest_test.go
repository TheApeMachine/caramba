package runtime

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestResolveManifestModelConfig(t *testing.T) {
	Convey("Given the embedded GPT-2 model manifest", t, func() {
		config, err := resolveManifestModelConfig(ModelConfig{
			Manifest: DefaultModelManifest,
		})

		Convey("It should resolve runtime, model, tokenizer, and generation seed", func() {
			So(err, ShouldBeNil)
			So(config.Manifest, ShouldEqual, DefaultModelManifest)
			So(config.Runtime, ShouldEqual, "model")
			So(config.Backend, ShouldEqual, "auto")
			So(config.Model, ShouldEqual, "openai-community/gpt2")
			So(config.Tokenizer, ShouldEqual, "openai-community/gpt2")
			So(config.ModelRepoType, ShouldEqual, "model")
			So(config.TokenizerRepoType, ShouldEqual, "model")
			So(config.Seed, ShouldEqual, int64(0))
		})
	})
}

func TestValidateModelRuntime(t *testing.T) {
	Convey("Given model runtime configuration", t, func() {
		Convey("It should accept the manifest-declared model runtime", func() {
			So(validateModelRuntime(ModelConfig{Runtime: "model"}), ShouldBeNil)
		})

		Convey("It should reject unrelated runtime names", func() {
			err := validateModelRuntime(ModelConfig{Runtime: "research-only"})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported manifest runtime")
		})
	})
}

func BenchmarkResolveManifestModelConfig(benchmark *testing.B) {
	for benchmark.Loop() {
		_, err := resolveManifestModelConfig(ModelConfig{
			Manifest: DefaultModelManifest,
		})

		if err != nil {
			benchmark.Fatal(err)
		}
	}
}
