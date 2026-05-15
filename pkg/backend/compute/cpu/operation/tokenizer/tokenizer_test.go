package tokenizer

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLoad_Forward(test *testing.T) {
	Convey("Given a tokenizer.load operation", test, func() {
		source := writeTokenizerFixture(test)
		operation := NewLoad()

		Convey("It should load the tokenizer and emit vocab size", func() {
			stateDict := state.NewDict()
			stateDict.Source = source

			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldResemble, []float64{12})
		})
	})
}

func TestEncode_Forward(test *testing.T) {
	Convey("Given a tokenizer.encode operation", test, func() {
		source := writeTokenizerFixture(test)
		operation := NewEncode()

		Convey("It should encode prompt text into token IDs", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Text = "hello hello!"

			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldResemble, []float64{7, 8, 9})
		})

		Convey("It should pad encoded token IDs when requested", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Text = "hello"
			stateDict.PadTo = 3
			stateDict.PadID = 10

			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Out, ShouldResemble, []float64{7, 10, 10})
		})
	})
}

func TestDecode_Forward(test *testing.T) {
	Convey("Given a tokenizer.decode operation", test, func() {
		source := writeTokenizerFixture(test)
		operation := NewDecode()

		Convey("It should decode token IDs into state text", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.WithInputs([]float64{7, 8, 9})

			outputState, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(outputState.Text, ShouldEqual, "hello hello!")
			So(outputState.Out, ShouldResemble, []float64{12})
		})
	})
}

func BenchmarkEncode_Forward(benchmark *testing.B) {
	source := writeTokenizerFixture(benchmark)
	operation := NewEncode()

	for benchmark.Loop() {
		stateDict := state.NewDict()
		stateDict.Source = source
		stateDict.Text = "hello hello!"
		_, _ = operation.Forward(stateDict)
	}
}

func writeTokenizerFixture(testingHandle interface {
	Helper()
	TempDir() string
	Fatalf(string, ...any)
}) string {
	testingHandle.Helper()

	source := testingHandle.TempDir()
	path := filepath.Join(source, "tokenizer.json")
	err := os.WriteFile(path, tokenizerFixtureJSON(), 0o644)

	if err != nil {
		testingHandle.Fatalf("write tokenizer fixture: %v", err)
	}

	return source
}

func tokenizerFixtureJSON() []byte {
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
      "special": true
    }
  ]
}`)
}
