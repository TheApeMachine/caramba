//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalMasking_ApplyMaskTensor(test *testing.T) {
	Convey("Given resident Metal apply-mask inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		maskingOps := maskingOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				shape, err := computetensor.NewShape([]int{elementCount})
				So(err, ShouldBeNil)

				scores := maskingScores(elementCount)
				mask := maskingMask(elementCount)
				expected := referenceApplyMask(scores, mask)
				scoreTensor := uploadMetalTensorForTest(test, tensorBackend, shape, scores)
				maskTensor := uploadMetalTensorForTest(test, tensorBackend, shape, mask)

				output, err := maskingOps.ApplyMaskTensor(scoreTensor, maskTensor)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMaskValues(values, expected, 1e-6)
			}
		})
	})
}

func TestMetalMasking_CausalMaskTensor(test *testing.T) {
	Convey("Given resident Metal causal-mask output shapes", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		maskingOps := maskingOpsForTest(test, tensorBackend)

		Convey("It should match the causal attention mask contract", func() {
			for _, seqLen := range []int{1, 7, 64} {
				outputShape, err := computetensor.NewShape([]int{seqLen, seqLen})
				So(err, ShouldBeNil)

				output, err := maskingOps.CausalMaskTensor(outputShape, seqLen)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertCausalMask(values, seqLen)
			}
		})
	})
}

func BenchmarkMetalMasking_ApplyMaskTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	maskingOps, err := tensorBackend.masking()
	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := computetensor.NewShape([]int{8192})
	if err != nil {
		benchmark.Fatal(err)
	}

	scores := uploadMetalTensor(tensorBackend, shape, maskingScores(shape.Len()))
	defer func() {
		_ = scores.Close()
	}()

	mask := uploadMetalTensor(tensorBackend, shape, maskingMask(shape.Len()))
	defer func() {
		_ = mask.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := maskingOps.ApplyMaskTensor(scores, mask)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkMetalMasking_CausalMaskTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	maskingOps, err := tensorBackend.masking()
	if err != nil {
		benchmark.Fatal(err)
	}

	outputShape, err := computetensor.NewShape([]int{128, 128})
	if err != nil {
		benchmark.Fatal(err)
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := maskingOps.CausalMaskTensor(outputShape, 128)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func maskingOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalMasking {
	test.Helper()

	maskingOps, err := tensorBackend.masking()
	So(err, ShouldBeNil)

	return maskingOps
}

func maskingScores(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64((index%23)-11) / 5
	}

	return values
}

func maskingMask(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		if index%11 == 0 {
			values[index] = math.Inf(-1)
			continue
		}

		values[index] = float64((index%7)-3) / 4
	}

	return values
}

func referenceApplyMask(scores []float64, mask []float64) []float64 {
	values := make([]float64, len(scores))

	for index, score := range scores {
		values[index] = float64(float32(float32(score) + float32(mask[index])))
	}

	return values
}

func assertMaskValues(actual []float64, expected []float64, tolerance float64) {
	So(actual, ShouldHaveLength, len(expected))

	for index, value := range actual {
		if math.IsInf(expected[index], -1) {
			SoMsg("mask value should be -Inf", math.IsInf(value, -1), ShouldBeTrue)
			continue
		}

		SoMsg("finite mask value", math.Abs(value-expected[index]) <= tolerance, ShouldBeTrue)
	}
}

func assertCausalMask(values []float64, seqLen int) {
	So(values, ShouldHaveLength, seqLen*seqLen)

	for rowIndex := range seqLen {
		for colIndex := range seqLen {
			value := values[rowIndex*seqLen+colIndex]
			if colIndex <= rowIndex {
				So(value, ShouldEqual, 0)
				continue
			}

			SoMsg("causal mask should be -Inf", math.IsInf(value, -1), ShouldBeTrue)
		}
	}
}
