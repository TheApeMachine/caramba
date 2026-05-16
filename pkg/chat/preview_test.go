package chat

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/tokenizer"
)

func TestPreviewGenerator_Generate(test *testing.T) {
	Convey("Given a preview generator without a tokenizer", test, func() {
		generator, err := NewPreviewGenerator(context.Background(), PreviewConfig{})

		So(err, ShouldBeNil)

		Convey("It should stream an explicit preview response", func() {
			output := bytes.NewBuffer(nil)
			err := generator.Generate(context.Background(), "hello", outputEmit(output))

			So(err, ShouldBeNil)
			So(output.String(), ShouldContainSubstring, "Preview runtime active")
			So(output.String(), ShouldContainSubstring, "manifest-backed local inference")
		})
	})

	Convey("Given a preview generator with a tokenizer", test, func() {
		tokenizerRoot := test.TempDir()
		tokenizerPath := filepath.Join(tokenizerRoot, "tokenizer.json")
		err := os.WriteFile(tokenizerPath, previewTokenizerJSON(), 0o644)

		So(err, ShouldBeNil)

		generator, err := NewPreviewGenerator(
			context.Background(),
			PreviewConfig{
				TokenizerSource: tokenizer.Source{Source: tokenizerRoot},
			},
		)

		So(err, ShouldBeNil)

		Convey("It should report prompt tokenization", func() {
			output := bytes.NewBuffer(nil)
			err := generator.Generate(context.Background(), "hello hello!", outputEmit(output))

			So(err, ShouldBeNil)
			So(output.String(), ShouldContainSubstring, "3 tokens")
		})
	})
}

func BenchmarkPreviewGenerator_Generate(benchmark *testing.B) {
	generator, err := NewPreviewGenerator(context.Background(), PreviewConfig{})

	if err != nil {
		benchmark.Fatal(err)
	}

	for benchmark.Loop() {
		output := bytes.NewBuffer(nil)
		_ = generator.Generate(context.Background(), "hello", outputEmit(output))
	}
}

func outputEmit(output *bytes.Buffer) func(string) error {
	return func(chunk string) error {
		_, err := output.WriteString(chunk)

		return err
	}
}

func previewTokenizerJSON() []byte {
	return []byte(`{
  "version": "1.0",
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "Ġ": 4,
      "he": 5,
      "hel": 6,
      "hello": 7,
      "Ġhello": 8,
      "!": 9
    },
    "merges": [
      "h e",
      "he l",
      "hel l",
      "hell o",
      "Ġ hello"
    ]
  },
  "decoder": {
    "type": "ByteLevel"
  },
  "added_tokens": []
}`)
}
