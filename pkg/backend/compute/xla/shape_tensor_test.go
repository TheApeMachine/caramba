//go:build cgo && xla

package xla

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestTensorBackend_ReshapeTensor(test *testing.T) {
	Convey("Given a resident XLA tensor reshape", test, func() {
		tensorBackend := newXLATensorBackendForTest(test)
		defer tensorBackend.Close()

		input := uploadXLATensor(
			test,
			tensorBackend,
			[]int{2, 3},
			[]float64{1, 2, 3, 4, 5, 6},
		)
		defer input.Close()
		outputShape := mustXLAShape(test, []int{3, 2})

		Convey("It should reshape without staging through host dispatch", func() {
			output, err := tensorBackend.ReshapeTensor(input, outputShape)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 2, 3, 4, 5, 6})
		})
	})
}

func TestTensorBackend_TransposeTensor(test *testing.T) {
	Convey("Given a resident XLA tensor transpose", test, func() {
		tensorBackend := newXLATensorBackendForTest(test)
		defer tensorBackend.Close()

		input := uploadXLATensor(
			test,
			tensorBackend,
			[]int{2, 3},
			[]float64{1, 2, 3, 4, 5, 6},
		)
		defer input.Close()
		outputShape := mustXLAShape(test, []int{3, 2})

		Convey("It should transpose without staging through host dispatch", func() {
			output, err := tensorBackend.TransposeTensor(input, outputShape, 0, 1)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 4, 2, 5, 3, 6})
		})
	})
}

func TestTensorBackend_ViewAsHeadsTensor(test *testing.T) {
	Convey("Given a resident XLA view_as_heads operation", test, func() {
		tensorBackend := newXLATensorBackendForTest(test)
		defer tensorBackend.Close()

		input := uploadXLATensor(
			test,
			tensorBackend,
			[]int{1, 2, 4},
			[]float64{1, 2, 3, 4, 5, 6, 7, 8},
		)
		defer input.Close()
		outputShape := mustXLAShape(test, []int{1, 2, 2, 2})

		Convey("It should move head storage through StableHLO", func() {
			output, err := tensorBackend.ViewAsHeadsTensor(input, outputShape, 1, 2, 2, 2)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 5, 6, 3, 4, 7, 8})
		})
	})
}

func TestTensorBackend_LastTokenTensor(test *testing.T) {
	Convey("Given a resident XLA last_token operation", test, func() {
		tensorBackend := newXLATensorBackendForTest(test)
		defer tensorBackend.Close()

		input := uploadXLATensor(
			test,
			tensorBackend,
			[]int{2, 3, 2},
			[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		)
		defer input.Close()
		outputShape := mustXLAShape(test, []int{2, 2})

		Convey("It should select the final sequence token through StableHLO", func() {
			output, err := tensorBackend.LastTokenTensor(input, outputShape, 2, 3, 2)
			So(err, ShouldBeNil)
			defer output.Close()

			values, err := tensorBackend.DownloadFloat64(output)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{5, 6, 11, 12})
		})
	})
}

func BenchmarkTensorBackend_ReshapeTensor(benchmark *testing.B) {
	tensorBackend := newXLATensorBackendForBenchmark(benchmark)
	defer tensorBackend.Close()

	input := uploadXLATensorForBenchmark(
		benchmark,
		tensorBackend,
		[]int{1, 8192},
		make([]float64, 8192),
	)
	defer input.Close()
	outputShape := mustXLAShape(benchmark, []int{8192, 1})

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := tensorBackend.ReshapeTensor(input, outputShape)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}
