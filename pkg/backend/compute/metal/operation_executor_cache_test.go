//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestTensorBackend_cachedTensor(test *testing.T) {
	Convey("Given a Metal resident parameter cache", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)

		Convey("It should reuse unchanged resident tensors", func() {
			first, err := tensorBackend.cachedTensor("linear:weight", []int{2}, []float64{1, 2})
			So(err, ShouldBeNil)

			second, err := tensorBackend.cachedTensor("linear:weight", []int{2}, []float64{1, 2})
			So(err, ShouldBeNil)

			So(second, ShouldEqual, first)
		})

		Convey("It should replace resident tensors when values change", func() {
			first, err := tensorBackend.cachedTensor("linear:weight", []int{2}, []float64{1, 2})
			So(err, ShouldBeNil)

			second, err := tensorBackend.cachedTensor("linear:weight", []int{2}, []float64{3, 4})
			So(err, ShouldBeNil)

			So(second, ShouldNotEqual, first)

			values, err := tensorFloat64Values(second)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{3, 4})
		})

		Convey("It should replace resident tensors when shape changes", func() {
			first, err := tensorBackend.cachedTensor("linear:weight", []int{2}, []float64{1, 2})
			So(err, ShouldBeNil)

			second, err := tensorBackend.cachedTensor("linear:weight", []int{1, 2}, []float64{1, 2})
			So(err, ShouldBeNil)

			So(second, ShouldNotEqual, first)
			So(second.Shape().Dims(), ShouldResemble, []int{1, 2})
		})
	})
}

func BenchmarkTensorBackend_cachedTensor(benchmark *testing.B) {
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	shape := []int{1024}
	values := make([]float64, 1024)

	for valueIndex := range values {
		values[valueIndex] = float64(valueIndex)
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := tensorBackend.cachedTensor("linear:weight", shape, values); err != nil {
			benchmark.Fatal(err)
		}
	}
}
