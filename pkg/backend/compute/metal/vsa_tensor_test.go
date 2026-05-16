//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalVSAOps_BindTensor(test *testing.T) {
	Convey("Given resident Metal VSA bind inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		vsaOps := vsaOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				shape := vsaShape(test, elementCount)
				left := vsaVector(elementCount, 0)
				right := vsaVector(elementCount, 5)
				leftTensor := uploadMetalTensorForTest(test, tensorBackend, shape, left)
				rightTensor := uploadMetalTensorForTest(test, tensorBackend, shape, right)

				output, err := vsaOps.BindTensor(leftTensor, rightTensor)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceVSABind(left, right), 1e-7)
			}
		})
	})
}

func TestMetalVSAOps_BundleTensor(test *testing.T) {
	Convey("Given resident Metal VSA bundle inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		vsaOps := vsaOpsForTest(test, tensorBackend)

		Convey("It should match the normalised float32 scalar reference", func() {
			for _, elementCount := range metalMathContractSizes {
				shape := vsaShape(test, elementCount)
				vectors := [][]float64{
					vsaVector(elementCount, 0),
					vsaVector(elementCount, 3),
					vsaVector(elementCount, 7),
				}
				flatTensor := uploadMetalTensorForTest(
					test,
					tensorBackend,
					vsaShape(test, len(vectors)*elementCount),
					flattenVSAVectors(vectors),
				)

				output, err := vsaOps.BundleTensor(flatTensor, shape, len(vectors))
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceVSABundle(vectors), 2e-5)
			}
		})
	})
}

func TestMetalVSAOps_SimilarityTensor(test *testing.T) {
	Convey("Given resident Metal VSA similarity inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		vsaOps := vsaOpsForTest(test, tensorBackend)

		Convey("It should match the float32 dot-product reference", func() {
			for _, elementCount := range metalMathContractSizes {
				shape := vsaShape(test, elementCount)
				left := vsaUnitVector(elementCount, 0.25)
				right := vsaUnitVector(elementCount, 0.5)
				leftTensor := uploadMetalTensorForTest(test, tensorBackend, shape, left)
				rightTensor := uploadMetalTensorForTest(test, tensorBackend, shape, right)

				output, err := vsaOps.SimilarityTensor(leftTensor, rightTensor, vsaShape(test, 1))
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, []float64{referenceVSASimilarity(left, right)}, 1e-6)
			}
		})
	})
}

func TestMetalVSAOps_PermuteTensor(test *testing.T) {
	Convey("Given resident Metal VSA permute inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		vsaOps := vsaOpsForTest(test, tensorBackend)

		Convey("It should match cyclic shift at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				shape := vsaShape(test, elementCount)
				input := vsaVector(elementCount, 2)
				inputTensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)

				output, err := vsaOps.PermuteTensor(inputTensor, 3)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceVSAPermute(input, 3), 1e-7)
			}
		})
	})
}

func TestMetalVSAOps_InversePermuteTensor(test *testing.T) {
	Convey("Given resident Metal VSA inverse-permute inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		vsaOps := vsaOpsForTest(test, tensorBackend)

		Convey("It should match inverse cyclic shift at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				shape := vsaShape(test, elementCount)
				input := vsaVector(elementCount, 4)
				inputTensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)

				output, err := vsaOps.InversePermuteTensor(inputTensor, 3)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, referenceVSAInversePermute(input, 3), 1e-7)
			}
		})
	})
}

func TestTensorBackend_applyVSAGraph(test *testing.T) {
	Convey("Given Metal VSA graph execution", test, func() {
		Convey("It should keep all VSA intermediates resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			shape := vsaShape(test, 64)
			left, right, third := vsaVector(64, 0), vsaVector(64, 5), vsaVector(64, 9)
			graph, targets := vsaGraph(shape, left, right, third)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(
				context.Background(),
				graph,
				targets,
			)
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, len(targets))
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64((len(left)+len(right)+len(third))*4))

			assertVSAGraphOutputs(results, left, right, third)
		})
	})
}

func BenchmarkMetalVSAOps_BindTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, vsaOps := vsaBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	shape := vsaBenchmarkShape(benchmark, 8192)
	left := uploadMetalTensor(tensorBackend, shape, vsaVector(shape.Len(), 0))
	right := uploadMetalTensor(tensorBackend, shape, vsaVector(shape.Len(), 5))
	defer closeBenchmarkTensors(left, right)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := vsaOps.BindTensor(left, right)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalVSAOps_BundleTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, vsaOps := vsaBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	shape := vsaBenchmarkShape(benchmark, 8192)
	vectors := [][]float64{vsaVector(shape.Len(), 0), vsaVector(shape.Len(), 3), vsaVector(shape.Len(), 7)}
	flat := uploadMetalTensor(
		tensorBackend,
		vsaBenchmarkShape(benchmark, len(vectors)*shape.Len()),
		flattenVSAVectors(vectors),
	)
	defer closeBenchmarkTensors(flat)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := vsaOps.BundleTensor(flat, shape, len(vectors))
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalVSAOps_SimilarityTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, vsaOps := vsaBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	shape := vsaBenchmarkShape(benchmark, 8192)
	left := uploadMetalTensor(tensorBackend, shape, vsaUnitVector(shape.Len(), 0.25))
	right := uploadMetalTensor(tensorBackend, shape, vsaUnitVector(shape.Len(), 0.5))
	outputShape := vsaBenchmarkShape(benchmark, 1)
	defer closeBenchmarkTensors(left, right)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := vsaOps.SimilarityTensor(left, right, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalVSAOps_PermuteTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, vsaOps := vsaBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	shape := vsaBenchmarkShape(benchmark, 8192)
	input := uploadMetalTensor(tensorBackend, shape, vsaVector(shape.Len(), 2))
	defer closeBenchmarkTensors(input)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := vsaOps.PermuteTensor(input, 3)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalVSAOps_InversePermuteTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, vsaOps := vsaBenchmarkOps(benchmark)
	defer func() {
		_ = tensorBackend.Close()
	}()

	shape := vsaBenchmarkShape(benchmark, 8192)
	input := uploadMetalTensor(tensorBackend, shape, vsaVector(shape.Len(), 2))
	defer closeBenchmarkTensors(input)

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := vsaOps.InversePermuteTensor(input, 3)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func vsaOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalVSAOps {
	test.Helper()

	vsaOps, err := tensorBackend.vsa()
	So(err, ShouldBeNil)

	return vsaOps
}

func vsaBenchmarkOps(benchmark *testing.B) (*TensorBackend, *MetalVSAOps) {
	benchmark.Helper()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	vsaOps, err := tensorBackend.vsa()
	if err != nil {
		benchmark.Fatal(err)
	}

	return tensorBackend, vsaOps
}

func vsaShape(test testing.TB, elementCount int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	So(err, ShouldBeNil)

	return shape
}

func vsaBenchmarkShape(benchmark *testing.B, elementCount int) computetensor.Shape {
	benchmark.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	if err != nil {
		benchmark.Fatal(err)
	}

	return shape
}
