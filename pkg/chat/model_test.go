package chat

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewModelGenerator(test *testing.T) {
	Convey("Given model runtime configuration", test, func() {
		Convey("It should require a model", func() {
			generator, err := NewModelGenerator(
				context.Background(),
				ModelConfig{MaxNewTokens: 16},
			)

			So(generator, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "model is required")
		})

		Convey("It should require a positive generation budget", func() {
			generator, err := NewModelGenerator(
				context.Background(),
				ModelConfig{Model: "openai-community/gpt2"},
			)

			So(generator, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "max_new_tokens")
		})

		Convey("It should fail explicitly until CausalLM is connected", func() {
			generator, err := NewModelGenerator(
				context.Background(),
				ModelConfig{
					Model:        "openai-community/gpt2",
					MaxNewTokens: 16,
				},
			)

			So(generator, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "CausalLM runtime is not connected")
		})
	})
}

func BenchmarkNewModelGenerator(benchmark *testing.B) {
	for benchmark.Loop() {
		_, _ = NewModelGenerator(
			context.Background(),
			ModelConfig{
				Model:        "openai-community/gpt2",
				MaxNewTokens: 16,
			},
		)
	}
}
