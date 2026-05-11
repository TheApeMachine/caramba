//go:build cgo && xla

package xla

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestXLAActivation_ReLU(test *testing.T) {
	Convey("Given an XLA activation runtime", test, func() {
		activation := newXLAActivation(test)
		defer activation.Shutdown()

		cases := []struct {
			name     string
			input    []float64
			expected []float64
		}{
			{"all negative", []float64{-3, -1, -0.1}, []float64{0, 0, 0}},
			{"all positive", []float64{0.1, 1, 5}, []float64{0.1, 1, 5}},
			{"all zeros", []float64{0, 0, 0}, []float64{0, 0, 0}},
			{"large magnitude", []float64{-1e6, 1e6}, []float64{0, 1e6}},
			{"near zero", []float64{-1e-12, 0, 1e-12}, []float64{0, 0, 1e-12}},
		}

		for _, testcase := range cases {
			Convey(testcase.name, func() {
				values, err := activation.ReLU(testcase.input)

				So(err, ShouldBeNil)
				So(values, ShouldResemble, testcase.expected)
			})
		}
	})
}

func BenchmarkXLAActivation_ReLU(benchmark *testing.B) {
	activation := newXLAActivation(benchmark)
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

func newXLAActivation(tb testing.TB) *XLAActivation {
	tb.Helper()

	platform := xlaPJRTAvailablePlatform(tb)
	activation, err := New(platform)

	if err != nil {
		tb.Skip(err)
	}

	return activation
}

func xlaPJRTAvailablePlatform(tb testing.TB) string {
	tb.Helper()

	for _, platform := range []string{"cpu", "gpu"} {
		config, err := NewPJRTConfig(platform)

		if err != nil {
			continue
		}

		if config.ValidateRuntime() == nil {
			return platform
		}
	}

	tb.Skip("xla activation: no PJRT plugin configured")

	return ""
}
