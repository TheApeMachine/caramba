package cmd

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestChatCommandOptions_TokenizerSource(test *testing.T) {
	Convey("Given chat command options", test, func() {
		Convey("It should prefer an explicit tokenizer source", func() {
			options := chatCommandOptions{
				Model:     "model/repo",
				Tokenizer: "tokenizer/repo",
				Revision:  "main",
				RepoType:  "model",
				Cache:     "/tmp/cache",
			}

			source := options.TokenizerSource()

			So(source.Source, ShouldEqual, "tokenizer/repo")
			So(source.Revision, ShouldEqual, "main")
			So(source.RepoType, ShouldEqual, "model")
			So(source.Cache, ShouldEqual, "/tmp/cache")
		})

		Convey("It should use the model source when tokenizer is omitted", func() {
			options := chatCommandOptions{
				Model:    "model/repo",
				RepoType: "model",
			}

			source := options.TokenizerSource()

			So(source.Source, ShouldEqual, "model/repo")
		})
	})
}

func BenchmarkChatCommandOptions_TokenizerSource(benchmark *testing.B) {
	options := chatCommandOptions{
		Model:    "openai-community/gpt2",
		RepoType: "model",
	}

	for benchmark.Loop() {
		_ = options.TokenizerSource()
	}
}
