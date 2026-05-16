package cmd

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/spf13/cobra"
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

func TestChatCommandOptions_RuntimeName(test *testing.T) {
	Convey("Given automatic chat runtime selection", test, func() {
		command := &cobra.Command{}
		command.Flags().String("model", "", "")
		command.Flags().String("manifest", "", "")

		Convey("It should use preview without model inputs", func() {
			options := chatCommandOptions{Runtime: "auto"}

			So(options.RuntimeName(command), ShouldEqual, "preview")
		})

		Convey("It should use model runtime when model is set", func() {
			options := chatCommandOptions{Runtime: "auto"}
			err := command.Flags().Set("model", "openai-community/gpt2")

			So(err, ShouldBeNil)
			So(options.RuntimeName(command), ShouldEqual, "model")
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
