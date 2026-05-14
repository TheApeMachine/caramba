package hub

import (
	"bytes"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestDecodeXorbRange(test *testing.T) {
	Convey("Given an uncompressed xorb range", test, func() {
		data := bytes.Join([][]byte{
			xorbChunk(0, []byte("hello ")),
			xorbChunk(0, []byte("world")),
		}, nil)

		Convey("It should return chunks indexed by their xorb position", func() {
			chunks, err := DecodeXorbRange(data, 3, 5)

			So(err, ShouldBeNil)
			So(chunks[3], ShouldResemble, []byte("hello "))
			So(chunks[4], ShouldResemble, []byte("world"))
		})
	})
}

func TestDecodeLZ4Block(test *testing.T) {
	Convey("Given an LZ4 literal-only block", test, func() {
		out, err := decodeLZ4Block([]byte{0x30, 'a', 'b', 'c'}, 3)

		Convey("It should decode the literal bytes", func() {
			So(err, ShouldBeNil)
			So(out, ShouldResemble, []byte("abc"))
		})
	})

	Convey("Given an LZ4 block with an overlapping match", test, func() {
		out, err := decodeLZ4Block([]byte{0x11, 'a', 0x01, 0x00}, 6)

		Convey("It should copy from the decoded window", func() {
			So(err, ShouldBeNil)
			So(out, ShouldResemble, []byte("aaaaaa"))
		})
	})
}

func BenchmarkDecodeXorbRange(benchmark *testing.B) {
	data := xorbChunk(0, bytes.Repeat([]byte("a"), 64*1024))

	for benchmark.Loop() {
		_, _ = DecodeXorbRange(data, 0, 1)
	}
}

func xorbChunk(compression byte, payload []byte) []byte {
	header := []byte{
		0,
		byte(len(payload)),
		byte(len(payload) >> 8),
		byte(len(payload) >> 16),
		compression,
		byte(len(payload)),
		byte(len(payload) >> 8),
		byte(len(payload) >> 16),
	}

	return append(header, payload...)
}
