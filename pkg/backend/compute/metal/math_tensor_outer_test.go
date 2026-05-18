//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMathOps_OuterTensor(test *testing.T) {
	Convey("Given resident Metal outer product inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				leftValues := []float64{1.5}
				rightValues := metalMathInput(elementCount, -2, 2)
				expected := referenceOuter(leftValues, rightValues)

				leftShape, err := computetensor.NewShape([]int{len(leftValues)})
				So(err, ShouldBeNil)

				rightShape, err := computetensor.NewShape([]int{len(rightValues)})
				So(err, ShouldBeNil)

				outputShape, err := computetensor.NewShape([]int{len(leftValues), len(rightValues)})
				So(err, ShouldBeNil)

				left := uploadMetalTensorForTest(test, tensorBackend, leftShape, leftValues)
				right := uploadMetalTensorForTest(test, tensorBackend, rightShape, rightValues)
				output, err := mathOps.OuterTensor(left, right, outputShape)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, expected, 1e-6)
			}
		})
	})
}

func TestMathOps_DropoutTensor(test *testing.T) {
	Convey("Given resident Metal dropout inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the seeded float32 scalar reference at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				input := metalMathInput(elementCount, -3, 3)
				expected := referenceDropout(input, 0.25, true, 17)

				output := runUnaryMathTensorForTest(
					test,
					tensorBackend,
					func(inputTensor computetensor.Tensor) (computetensor.Tensor, error) {
						return mathOps.DropoutTensor(inputTensor, 0.25, true, 17)
					},
					input,
				)
				assertMetalMaxDiff(output, expected, 1e-6)
			}
		})

		Convey("It should copy input when training is disabled", func() {
			input := metalMathInput(64, -1, 1)

			output := runUnaryMathTensorForTest(
				test,
				tensorBackend,
				func(inputTensor computetensor.Tensor) (computetensor.Tensor, error) {
					return mathOps.DropoutTensor(inputTensor, 0.5, false, 17)
				},
				input,
			)
			assertMetalMaxDiff(output, mapFloat32Reference(input, identityReference), 0)
		})
	})
}

func BenchmarkMathOps_OuterTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	mathOps, err := tensorBackend.math()
	if err != nil {
		benchmark.Fatal(err)
	}

	leftShape, rightShape, outputShape := mustOuterBenchmarkShapes()
	left := uploadMetalTensor(tensorBackend, leftShape, metalMathInput(leftShape.Len(), -1, 1))
	right := uploadMetalTensor(tensorBackend, rightShape, metalMathInput(rightShape.Len(), -1, 1))
	defer func() {
		_ = left.Close()
		_ = right.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := mathOps.OuterTensor(left, right, outputShape)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkMathOps_DropoutTensor(benchmark *testing.B) {
	benchmarkUnaryMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return func(input computetensor.Tensor) (computetensor.Tensor, error) {
			return mathOps.DropoutTensor(input, 0.25, true, 17)
		}
	})
}

func referenceOuter(left, right []float64) []float64 {
	output := make([]float64, len(left)*len(right))

	for row, leftValue := range left {
		for column, rightValue := range right {
			output[row*len(right)+column] = float64(float32(float32(leftValue) * float32(rightValue)))
		}
	}

	return output
}

func referenceDropout(
	input []float64,
	probability float64,
	training bool,
	seed int,
) []float64 {
	output := make([]float64, len(input))
	probability32 := float32(probability)

	for index, value := range input {
		if !training || probability32 == 0 {
			output[index] = float64(float32(value))
			continue
		}

		unit := dropoutUnit(uint64(seed), uint64(index))
		if unit < probability32 {
			continue
		}

		output[index] = float64(float32(float32(value) / (1 - probability32)))
	}

	return output
}

func dropoutUnit(seed uint64, index uint64) float32 {
	mixed := dropoutMix64((seed << 32) ^ index)

	return float32(mixed>>11) * float32(1.0/9007199254740992.0)
}

func dropoutMix64(value uint64) uint64 {
	value ^= value >> 30
	value *= 0xbf58476d1ce4e5b9
	value ^= value >> 27
	value *= 0x94d049bb133111eb
	value ^= value >> 31

	return value
}

func identityReference(value float64) float64 {
	return value
}

func mustOuterBenchmarkShapes() (computetensor.Shape, computetensor.Shape, computetensor.Shape) {
	leftShape, err := computetensor.NewShape([]int{64})
	if err != nil {
		panic(err)
	}

	rightShape, err := computetensor.NewShape([]int{128})
	if err != nil {
		panic(err)
	}

	outputShape, err := computetensor.NewShape([]int{64, 128})
	if err != nil {
		panic(err)
	}

	return leftShape, rightShape, outputShape
}
