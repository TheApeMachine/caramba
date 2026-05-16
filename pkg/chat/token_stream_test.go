package chat

import (
	"testing"

	"github.com/theapemachine/caramba/pkg/tokenizer"

	. "github.com/smartystreets/goconvey/convey"
)

func TestTokenStream_Append(test *testing.T) {
	Convey("Given generated token IDs that split a UTF-8 byte sequence", test, func() {
		artifact, err := tokenizer.Parse(testTokenizerJSON())
		So(err, ShouldBeNil)

		stream := newTokenStream(artifact.Tokenizer)
		chunks := make([]string, 0)
		emit := func(chunk string) error {
			chunks = append(chunks, chunk)

			return nil
		}

		Convey("It should wait until the buffered tokens decode to valid UTF-8", func() {
			err = stream.Append([]int{10}, emit)
			So(err, ShouldBeNil)
			So(chunks, ShouldBeEmpty)

			err = stream.Append([]int{11}, emit)
			So(err, ShouldBeNil)
			So(chunks, ShouldResemble, []string{"é"})
		})
	})
}

func TestTokenStream_Flush(test *testing.T) {
	Convey("Given an incomplete generated UTF-8 byte sequence", test, func() {
		artifact, err := tokenizer.Parse(testTokenizerJSON())
		So(err, ShouldBeNil)

		stream := newTokenStream(artifact.Tokenizer)
		emit := func(string) error { return nil }

		err = stream.Append([]int{10}, emit)
		So(err, ShouldBeNil)

		Convey("It should report the incomplete sequence at the stream boundary", func() {
			err = stream.Flush(emit)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "incomplete tokenizer byte sequence")
		})
	})
}

func BenchmarkTokenStream_Append(benchmark *testing.B) {
	artifact, err := tokenizer.Parse(testTokenizerJSON())

	if err != nil {
		benchmark.Fatal(err)
	}

	stream := newTokenStream(artifact.Tokenizer)
	emit := func(string) error { return nil }

	for benchmark.Loop() {
		_ = stream.Append([]int{7}, emit)
	}
}
