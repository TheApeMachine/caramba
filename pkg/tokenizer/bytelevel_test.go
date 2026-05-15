package tokenizer

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestByteLevelBPE_Encode(test *testing.T) {
	Convey("Given a byte-level BPE tokenizer", test, func() {
		artifact, err := Parse(testTokenizerJSON())

		So(err, ShouldBeNil)

		tokenizer := artifact.Tokenizer

		Convey("It should encode text into merged token IDs", func() {
			tokenIDs, err := tokenizer.Encode("hello hello!")

			So(err, ShouldBeNil)
			So(tokenIDs, ShouldResemble, []int{7, 8, 9})
		})

		Convey("It should preserve special tokens as atomic IDs", func() {
			tokenIDs, err := tokenizer.Encode("<|endoftext|>hello")

			So(err, ShouldBeNil)
			So(tokenIDs, ShouldResemble, []int{10, 7})
		})

		Convey("It should preserve repeated leading spaces", func() {
			tokenIDs, err := tokenizer.Encode("  hello")

			So(err, ShouldBeNil)
			So(tokenIDs, ShouldResemble, []int{4, 8})
		})
	})
}

func TestByteLevelBPE_Decode(test *testing.T) {
	Convey("Given byte-level BPE token IDs", test, func() {
		artifact, err := Parse(testTokenizerJSON())

		So(err, ShouldBeNil)

		tokenizer := artifact.Tokenizer

		Convey("It should decode IDs back to text", func() {
			text, err := tokenizer.Decode([]int{7, 8, 9}, false)

			So(err, ShouldBeNil)
			So(text, ShouldEqual, "hello hello!")
		})

		Convey("It should optionally skip special tokens", func() {
			text, err := tokenizer.Decode([]int{10, 7}, true)

			So(err, ShouldBeNil)
			So(text, ShouldEqual, "hello")
		})
	})
}

func TestRead(test *testing.T) {
	Convey("Given a tokenizer.json file", test, func() {
		path := filepath.Join(test.TempDir(), "tokenizer.json")
		err := os.WriteFile(path, testTokenizerJSON(), 0o644)

		So(err, ShouldBeNil)

		Convey("It should parse the tokenizer artifact", func() {
			artifact, err := Read(path)

			So(err, ShouldBeNil)
			So(artifact.Backend, ShouldEqual, "bytelevel_bpe")
			So(artifact.Tokenizer.VocabSize(), ShouldEqual, 12)
		})
	})
}

func BenchmarkByteLevelBPE_Encode(benchmark *testing.B) {
	artifact, err := Parse(testTokenizerJSON())

	if err != nil {
		benchmark.Fatal(err)
	}

	for benchmark.Loop() {
		_, _ = artifact.Tokenizer.Encode("hello hello!")
	}
}

func testTokenizerJSON() []byte {
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
      "!": 9,
      "<|endoftext|>": 10,
      "hell": 11
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
  "added_tokens": [
    {
      "id": 10,
      "content": "<|endoftext|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ]
}`)
}
