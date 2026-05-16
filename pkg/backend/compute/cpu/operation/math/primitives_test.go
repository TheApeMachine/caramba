package math

import (
	stdmath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestReduceSum(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("reduceSum", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)

					So(reduceSum(input), ShouldAlmostEqual, primitiveReferenceSum(input), 1e-9)
				}
			})
		})
	})
}

func TestReduceMax(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("reduceMax", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				So(reduceMax(nil), ShouldEqual, -stdmath.MaxFloat64)

				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)

					So(reduceMax(input), ShouldAlmostEqual, primitiveReferenceMax(input), 1e-12)
				}
			})
		})
	})
}

func TestDivScalar(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("divScalar", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)
					output := append([]float64(nil), input...)
					divisor := 3.25

					divScalar(output, divisor)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, input[elementIndex]/divisor, 1e-12)
					}
				}
			})
		})
	})
}

func TestAddVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("addVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					left, right := primitiveParityVectors(elementCount)
					output := make([]float64, elementCount)

					addVec(output, left, right)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, left[elementIndex]+right[elementIndex], 1e-12)
					}
				}
			})
		})
	})
}

func TestMulVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("mulVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					left, right := primitiveParityVectors(elementCount)
					output := make([]float64, elementCount)

					mulVec(output, left, right)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, left[elementIndex]*right[elementIndex], 1e-12)
					}
				}
			})
		})
	})
}

func TestMulScalar(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("mulScalar", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)
					output := append([]float64(nil), input...)
					scale := -1.75

					mulScalar(output, scale)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, input[elementIndex]*scale, 1e-12)
					}
				}
			})
		})
	})
}

func TestReduceSumSq(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("reduceSumSq", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)

					So(reduceSumSq(input), ShouldAlmostEqual, primitiveReferenceSumSq(input), 1e-9)
				}
			})
		})
	})
}

func TestSignVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("signVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)

					if elementCount > 2 {
						input[1] = 0
						input[2] = stdmath.NaN()
					}

					output := make([]float64, elementCount)
					signVec(output, input)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldEqual, primitiveReferenceSign(input[elementIndex]))
					}
				}
			})
		})
	})
}

func TestOuterRow(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("outerRow", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)
					output := make([]float64, elementCount)
					scale := 0.375

					outerRow(output, input, scale)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, input[elementIndex]*scale, 1e-12)
					}
				}
			})
		})
	})
}

func TestAddScaledVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("addScaledVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, source := primitiveParityVectors(elementCount)
					output := append([]float64(nil), input...)
					scale := -0.375

					addScaledVec(output, source, scale)

					for elementIndex := range elementCount {
						expected := input[elementIndex] + source[elementIndex]*scale
						So(output[elementIndex], ShouldAlmostEqual, expected, 1e-12)
					}
				}
			})
		})
	})
}

func TestSqrtVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("sqrtVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input := make([]float64, elementCount)
					output := make([]float64, elementCount)

					for elementIndex := range elementCount {
						input[elementIndex] = float64(elementIndex+1) * 0.125
					}

					sqrtVec(output, input)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, stdmath.Sqrt(input[elementIndex]), 1e-12)
					}
				}
			})
		})
	})
}

func TestAddScalarVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("addScalarVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)
					output := append([]float64(nil), input...)
					scalar := -0.8125

					addScalarVec(output, scalar)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, input[elementIndex]+scalar, 1e-12)
					}
				}
			})
		})
	})
}

func TestDivVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("divVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					left, right := primitiveParityVectors(elementCount)
					output := make([]float64, elementCount)

					for elementIndex := range right {
						right[elementIndex] += 3.0
					}

					divVec(output, left, right)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, left[elementIndex]/right[elementIndex], 1e-12)
					}
				}
			})
		})
	})
}

func TestClampVec(test *testing.T) {
	Convey("Given math primitives", test, func() {
		Convey("clampVec", func() {
			Convey("It should match scalar parity across SIMD lengths", func() {
				for _, elementCount := range primitiveParityLengths() {
					input, _ := primitiveParityVectors(elementCount)
					output := append([]float64(nil), input...)
					low := -0.375
					high := 0.625

					clampVec(output, low, high)

					for elementIndex := range elementCount {
						So(output[elementIndex], ShouldAlmostEqual, primitiveReferenceClamp(input[elementIndex], low, high), 1e-12)
					}
				}
			})
		})
	})
}

func BenchmarkReduceSum(benchmark *testing.B) {
	input, _ := primitiveParityVectors(1 << 20)
	sum := 0.0

	for benchmark.Loop() {
		sum += reduceSum(input)
	}

	if sum == 0 {
		benchmark.Fatal("reduceSum returned zero")
	}
}

func BenchmarkAddScaledVec(benchmark *testing.B) {
	output, input := primitiveParityVectors(1 << 20)

	for benchmark.Loop() {
		addScaledVec(output, input, 0.03125)
	}
}

func primitiveParityLengths() []int {
	return []int{1, 7, 64, 1024, 8192}
}

func primitiveParityVectors(elementCount int) ([]float64, []float64) {
	left := make([]float64, elementCount)
	right := make([]float64, elementCount)

	for elementIndex := range elementCount {
		left[elementIndex] = float64(elementIndex%17-8) * 0.125
		right[elementIndex] = float64(elementIndex%11-5) * 0.0625
	}

	return left, right
}

func primitiveReferenceSum(input []float64) float64 {
	sum := 0.0

	for _, value := range input {
		sum += value
	}

	return sum
}

func primitiveReferenceMax(input []float64) float64 {
	maxValue := -stdmath.MaxFloat64

	for _, value := range input {
		if value > maxValue {
			maxValue = value
		}
	}

	return maxValue
}

func primitiveReferenceSumSq(input []float64) float64 {
	sum := 0.0

	for _, value := range input {
		sum += value * value
	}

	return sum
}

func primitiveReferenceSign(value float64) float64 {
	if value > 0 {
		return 1
	}

	if value < 0 {
		return -1
	}

	return 0
}

func primitiveReferenceClamp(value float64, low float64, high float64) float64 {
	if value < low {
		return low
	}

	if value > high {
		return high
	}

	return value
}
