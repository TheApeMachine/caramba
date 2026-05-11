//go:build cgo && xla

package xla

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestXLAActivation_ReLU(test *testing.T) {
	Convey("Given an XLA activation runtime", test, func() {
		activation := newXLAActivationForTest(test)
		defer activation.Shutdown()

		Convey("It should execute ReLU through PJRT", func() {
			values, err := activation.ReLU([]float64{-2, -0.5, 0, 3})
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{0, 0, 0, 3})
		})
	})
}

func BenchmarkXLAActivation_ReLU(benchmark *testing.B) {
	activation := newXLAActivationForBenchmark(benchmark)
	defer activation.Shutdown()

	values := make([]float64, 1024)

	for index := range values {
		values[index] = float64(index) - 512
	}

	benchmark.ResetTimer()

	for range benchmark.N {
		if _, err := activation.ReLU(values); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func newXLAActivationForTest(test *testing.T) *XLAActivation {
	test.Helper()

	platform := xlaActivationTestPlatform(test)
	activation, err := New(platform)

	if err != nil {
		test.Skip(err)
	}

	return activation
}

func newXLAActivationForBenchmark(benchmark *testing.B) *XLAActivation {
	benchmark.Helper()

	platform := xlaActivationBenchmarkPlatform(benchmark)
	activation, err := New(platform)

	if err != nil {
		benchmark.Skip(err)
	}

	return activation
}

func xlaActivationTestPlatform(test *testing.T) string {
	test.Helper()

	for _, platform := range []string{"cpu", "gpu"} {
		if NewPJRTConfig(platform).ValidateRuntime() == nil {
			return platform
		}
	}

	test.Skip("xla activation: no PJRT plugin configured")
	return ""
}

func xlaActivationBenchmarkPlatform(benchmark *testing.B) string {
	benchmark.Helper()

	for _, platform := range []string{"cpu", "gpu"} {
		if NewPJRTConfig(platform).ValidateRuntime() == nil {
			return platform
		}
	}

	benchmark.Skip("xla activation: no PJRT plugin configured")
	return ""
}
