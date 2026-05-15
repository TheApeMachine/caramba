package tokenizer

import (
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestParse(test *testing.T) {
	Convey("Given BPE tokenizer JSON without an explicit model type", test, func() {
		data := []byte(strings.Replace(
			string(testTokenizerJSON()),
			`"type": "BPE",`,
			"",
			1,
		))

		Convey("It should infer the BPE backend from vocab and merges", func() {
			artifact, err := Parse(data)

			So(err, ShouldBeNil)
			So(artifact.Backend, ShouldEqual, "bytelevel_bpe")
		})
	})
}

func BenchmarkParse(benchmark *testing.B) {
	data := testTokenizerJSON()

	for benchmark.Loop() {
		_, _ = Parse(data)
	}
}
