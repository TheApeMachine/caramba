package cmd

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestChatCommandOptions_ModelConfig(test *testing.T) {
	Convey("Given chat command options", test, func() {
		Convey("It should only project the selected manifest into model config", func() {
			options := chatCommandOptions{
				Manifest: " model/llm/gpt2.yml ",
			}

			config := options.ModelConfig()

			So(config.Manifest, ShouldEqual, "model/llm/gpt2.yml")
			So(config.Model, ShouldBeBlank)
			So(config.Tokenizer, ShouldBeBlank)
			So(config.Backend, ShouldBeBlank)
		})
	})
}

func BenchmarkChatCommandOptions_ModelConfig(benchmark *testing.B) {
	options := chatCommandOptions{
		Manifest: "model/llm/gpt2.yml",
	}

	for benchmark.Loop() {
		_ = options.ModelConfig()
	}
}
