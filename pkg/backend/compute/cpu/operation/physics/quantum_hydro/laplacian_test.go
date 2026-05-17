package quantum_hydro

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLaplacianForwardValidation(test *testing.T) {
	Convey("Given a Laplacian operator", test, func() {
		laplacian := NewLaplacian(0.5)

		Convey("It rejects missing inputs", func() {
			_, err := laplacian.Forward(state.NewDict())
			So(err, ShouldNotBeNil)
		})

		Convey("It rejects rank-4 input shapes", func() {
			stateDict := state.NewDict().WithInput([]float64{1, 2, 3, 4}).WithShape([]int{1, 1, 2, 2})

			_, err := laplacian.Forward(stateDict)
			So(err, ShouldNotBeNil)
		})

		Convey("It rejects shape/length mismatches", func() {
			stateDict := state.NewDict().WithInput([]float64{1, 2, 3, 4}).WithShape([]int{3, 2})

			_, err := laplacian.Forward(stateDict)
			So(err, ShouldNotBeNil)
		})
	})
}

func TestLaplacian1DParity(test *testing.T) {
	Convey("Given the 1D Laplacian kernel", test, func() {
		Convey("It matches the scalar reference at canonical SIMD lengths", func() {
			for _, length := range laplacianParityLengths() {
				src := laplacianMakeField1D(length)
				expected := make([]float64, length)
				actual := make([]float64, length)
				invH2 := laplacianInvH2(0.0625)

				laplacianScalar(expected, src, []int{length}, invH2)
				laplacianKernel(actual, src, []int{length}, invH2)

				for index := range length {
					So(actual[index], ShouldAlmostEqual, expected[index], 1e-12)
				}
			}
		})
	})
}

func TestLaplacian2DParity(test *testing.T) {
	Convey("Given the 2D Laplacian kernel", test, func() {
		Convey("It matches the scalar reference at canonical 2D shapes", func() {
			for _, shape := range laplacianParityShapes2D() {
				length := shape[0] * shape[1]
				src := laplacianMakeFieldND(shape)
				expected := make([]float64, length)
				actual := make([]float64, length)
				invH2 := laplacianInvH2(0.125)

				laplacianScalar(expected, src, shape, invH2)
				laplacianKernel(actual, src, shape, invH2)

				for index := range length {
					So(actual[index], ShouldAlmostEqual, expected[index], 1e-12)
				}
			}
		})
	})
}

func TestLaplacian3DParity(test *testing.T) {
	Convey("Given the 3D Laplacian kernel", test, func() {
		Convey("It matches the scalar reference at canonical 3D shapes", func() {
			for _, shape := range laplacianParityShapes3D() {
				length := shape[0] * shape[1] * shape[2]
				src := laplacianMakeFieldND(shape)
				expected := make([]float64, length)
				actual := make([]float64, length)
				invH2 := laplacianInvH2(0.25)

				laplacianScalar(expected, src, shape, invH2)
				laplacianKernel(actual, src, shape, invH2)

				for index := range length {
					So(actual[index], ShouldAlmostEqual, expected[index], 1e-12)
				}
			}
		})
	})
}

func TestLaplacianScalarAnalyticConvergence(test *testing.T) {
	Convey("Given the analytic field u(x) = sin(2*pi*x) on the unit torus", test, func() {
		Convey("The discrete 1D Laplacian converges to -(2*pi)^2 * u at O(h^2)", func() {
			errors := []float64{}

			for _, n := range []int{64, 128, 256, 512, 1024} {
				h := 1.0 / float64(n)
				invH2 := 1.0 / (h * h)
				src := make([]float64, n)
				out := make([]float64, n)

				for index := range n {
					src[index] = stdmath.Sin(2.0 * stdmath.Pi * float64(index) * h)
				}

				laplacianScalar(out, src, []int{n}, invH2)

				worstError := 0.0
				expectedCoeff := -(2.0 * stdmath.Pi) * (2.0 * stdmath.Pi)

				for index := range n {
					expected := expectedCoeff * src[index]
					diff := stdmath.Abs(out[index] - expected)

					if diff > worstError {
						worstError = diff
					}
				}

				errors = append(errors, worstError)
			}

			Convey("Each refinement reduces the error by ~4x", func() {
				for index := 1; index < len(errors); index++ {
					ratio := errors[index-1] / errors[index]
					So(ratio, ShouldBeGreaterThan, 3.5)
					So(ratio, ShouldBeLessThan, 4.5)
				}
			})
		})
	})
}

func TestLaplacianForwardEndToEnd(test *testing.T) {
	Convey("Given the Forward method on a 2D grid", test, func() {
		spacing := 0.5
		shape := []int{8, 8}
		length := shape[0] * shape[1]
		src := laplacianMakeFieldND(shape)

		stateDict := state.NewDict().WithInput(src).WithShape(shape)
		laplacian := NewLaplacian(spacing)

		result, err := laplacian.Forward(stateDict)

		Convey("It returns no error and the expected output shape", func() {
			So(err, ShouldBeNil)
			So(len(result.Out), ShouldEqual, length)
		})

		Convey("It produces the same values as the scalar reference", func() {
			expected := make([]float64, length)
			laplacianScalar(expected, src, shape, laplacianInvH2(spacing))

			for index := range length {
				So(result.Out[index], ShouldAlmostEqual, expected[index], 1e-12)
			}
		})
	})
}

func BenchmarkLaplacian1D8192(benchmark *testing.B) {
	length := 8192
	src := laplacianMakeField1D(length)
	out := make([]float64, length)
	invH2 := laplacianInvH2(0.0625)
	shape := []int{length}

	for benchmark.Loop() {
		laplacianKernel(out, src, shape, invH2)
	}
}

func BenchmarkLaplacian2D128x128(benchmark *testing.B) {
	shape := []int{128, 128}
	length := shape[0] * shape[1]
	src := laplacianMakeFieldND(shape)
	out := make([]float64, length)
	invH2 := laplacianInvH2(0.0625)

	for benchmark.Loop() {
		laplacianKernel(out, src, shape, invH2)
	}
}

func BenchmarkLaplacian3D32x32x32(benchmark *testing.B) {
	shape := []int{32, 32, 32}
	length := shape[0] * shape[1] * shape[2]
	src := laplacianMakeFieldND(shape)
	out := make([]float64, length)
	invH2 := laplacianInvH2(0.0625)

	for benchmark.Loop() {
		laplacianKernel(out, src, shape, invH2)
	}
}

/*
laplacianParityLengths mirrors the canonical SIMD parity sweep used across
the math primitives package. The values exercise scalar-only (N=1), tail-
remainder (N=7), one-vector (N=64), multi-vector (N=1024), and large
sweep (N=8192) paths.
*/
func laplacianParityLengths() []int {
	return []int{1, 7, 64, 1024, 8192}
}

/*
laplacianParityShapes2D picks 2D grid sizes that exercise:
  - degenerate 1x1 (single cell wraps to itself)
  - tiny non-square (forces row-by-row scalar tail handling)
  - aligned 8x8 (one AVX2 vector per row)
  - large 32x32 (multi-vector rows, multi-row interior)
  - rectangular 64x128 (asymmetric strides, ~8k elements)
*/
func laplacianParityShapes2D() [][]int {
	return [][]int{
		{1, 1},
		{3, 7},
		{8, 8},
		{32, 32},
		{64, 128},
	}
}

/*
laplacianParityShapes3D picks 3D grid sizes that exercise:
  - degenerate 1x1x1
  - tiny non-cubic
  - cubic alignment
  - asymmetric depth-major
  - large near-8k cube-of-rectangles
*/
func laplacianParityShapes3D() [][]int {
	return [][]int{
		{1, 1, 1},
		{2, 3, 7},
		{4, 4, 4},
		{8, 8, 16},
		{8, 16, 64},
	}
}

/*
laplacianMakeField1D produces a deterministic 1D test field built from a
cheap rational sequence — bit-stable across architectures so parity tests
do not accumulate platform-specific noise from a transcendental generator.
*/
func laplacianMakeField1D(length int) []float64 {
	src := make([]float64, length)

	for index := range length {
		src[index] = float64(index%17-8)*0.125 + float64(index%5)*0.0625
	}

	return src
}

/*
laplacianMakeFieldND fills an arbitrary-rank field with the same
deterministic rational pattern, indexed by the flat element position.
*/
func laplacianMakeFieldND(shape []int) []float64 {
	length := 1

	for _, dim := range shape {
		length *= dim
	}

	return laplacianMakeField1D(length)
}

func laplacianInvH2(h float64) float64 { return 1.0 / (h * h) }
