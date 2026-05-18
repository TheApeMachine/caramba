package state

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestDict_float64Values(test *testing.T) {
	Convey("Given a state dictionary receiving dtype-aware tensors", test, func() {
		backend := tensor.NewHostBackend()
		shape, err := tensor.NewShape([]int{3})
		So(err, ShouldBeNil)

		input, err := backend.Upload(
			shape,
			dtype.Float32,
			dtypeconvert.Float32ToBytes([]float32{1.25, -2.5, 3.75}),
		)
		So(err, ShouldBeNil)
		defer input.Close()

		dict := NewDict(backend)

		Convey("It should convert through explicit raw bytes", func() {
			values, outputShape, err := dict.float64Values(input)

			So(err, ShouldBeNil)
			So(outputShape.Equal(shape), ShouldBeTrue)
			So(values, ShouldResemble, []float64{1.25, -2.5, 3.75})
		})
	})
}

func TestDict_RoPELayout(test *testing.T) {
	Convey("Given a RoPE state dictionary", test, func() {
		Convey("It should resolve head-major RoPE dimensions", func() {
			dict := NewDict().WithShape([]int{2, 4, 8, 16})
			dict.HeadDim = 16

			batch, numHeads, sequenceLength, headDim, err := dict.RoPELayout("rope")

			So(err, ShouldBeNil)
			So(batch, ShouldEqual, 2)
			So(numHeads, ShouldEqual, 4)
			So(sequenceLength, ShouldEqual, 8)
			So(headDim, ShouldEqual, 16)
		})

		Convey("It should reject rank-three projection tensors before head shaping", func() {
			dict := NewDict().WithShape([]int{1, 8, 64})
			dict.HeadDim = 16

			_, _, _, _, err := dict.RoPELayout("rope")

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "expected [batch, num_heads, seq_len, head_dim]")
		})

		Convey("It should reject odd RoPE head dimensions", func() {
			dict := NewDict().WithShape([]int{1, 2, 4, 15})
			dict.HeadDim = 15

			_, _, _, _, err := dict.RoPELayout("rope")

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "expected even head_dim")
		})

		Convey("It should reject configured head dimensions that disagree with shape", func() {
			dict := NewDict().WithShape([]int{1, 2, 4, 16})
			dict.HeadDim = 8

			_, _, _, _, err := dict.RoPELayout("rope")

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "does not match shape head dim")
		})
	})
}

func TestDict_GQALayout(test *testing.T) {
	Convey("Given a GQA state dictionary", test, func() {
		Convey("It should resolve rank-four query shape with configured KV heads", func() {
			dict := NewDict().WithShape([]int{1, 32, 7, 64})
			dict.NumHeads = 32
			dict.NumKVHeads = 8
			dict.HeadDim = 64

			batch, numHeads, numKVHeads, sequenceLength, headDim, err := dict.GQALayout("gqa")

			So(err, ShouldBeNil)
			So(batch, ShouldEqual, 1)
			So(numHeads, ShouldEqual, 32)
			So(numKVHeads, ShouldEqual, 8)
			So(sequenceLength, ShouldEqual, 7)
			So(headDim, ShouldEqual, 64)
		})

		Convey("It should keep supporting legacy rank-five GQA shape", func() {
			dict := NewDict().WithShape([]int{1, 32, 8, 7, 64})

			_, numHeads, numKVHeads, _, _, err := dict.GQALayout("gqa")

			So(err, ShouldBeNil)
			So(numHeads, ShouldEqual, 32)
			So(numKVHeads, ShouldEqual, 8)
		})

		Convey("It should reject non-divisible grouped heads", func() {
			dict := NewDict().WithShape([]int{1, 10, 7, 64})
			dict.NumHeads = 10
			dict.NumKVHeads = 4
			dict.HeadDim = 64

			_, _, _, _, _, err := dict.GQALayout("gqa")

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "num_heads must be divisible")
		})

		Convey("It should reject configured heads that disagree with shape", func() {
			dict := NewDict().WithShape([]int{1, 32, 7, 64})
			dict.NumHeads = 16
			dict.NumKVHeads = 8
			dict.HeadDim = 64

			_, _, _, _, _, err := dict.GQALayout("gqa")

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "does not match shape heads")
		})
	})
}

func BenchmarkDict_RoPELayout(benchmark *testing.B) {
	dict := NewDict().WithShape([]int{1, 32, 128, 64})
	dict.HeadDim = 64

	for benchmark.Loop() {
		_, _, _, _, _ = dict.RoPELayout("rope")
	}
}

func BenchmarkDict_GQALayout(benchmark *testing.B) {
	dict := NewDict().WithShape([]int{1, 32, 128, 64})
	dict.NumKVHeads = 8
	dict.HeadDim = 64

	for benchmark.Loop() {
		_, _, _, _, _, _ = dict.GQALayout("gqa")
	}
}
