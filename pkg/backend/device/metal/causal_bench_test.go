package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunCausalDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalCausalDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkCausalDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkCausalDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range metalCausalBenchmarkNames() {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			tensors := benchmarkCausalTensors(benchmark, backend, name, storageDType)
			defer closeBenchmarkTensors(tensors...)

			benchmark.SetBytes(causalBenchmarkBytes(name, storageDType))
			benchmark.ResetTimer()

			for benchmark.Loop() {
				if err := benchmarkCausalKernel(benchmark, name, storageDType).Run(tensors...); err != nil {
					benchmark.Fatal(err)
				}

				if err := tensors[len(tensors)-1].Sync(context.Background()); err != nil {
					benchmark.Fatal(err)
				}
			}
		})
	}
}

func metalCausalBenchmarkNames() []string {
	return []string{
		"backdoor_adjustment",
		"frontdoor_adjustment",
		"do_intervene",
		"cate",
		"counterfactual",
		"iv_estimate",
		"dag_markov_factorization",
	}
}

func benchmarkCausalTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) []tensor.Tensor {
	switch name {
	case "backdoor_adjustment":
		return benchmarkBackdoorTensors(testingObject, backend, storageDType)
	case "frontdoor_adjustment":
		return benchmarkFrontdoorTensors(testingObject, backend, storageDType)
	case "do_intervene":
		return benchmarkDoInterveneTensors(testingObject, backend, storageDType)
	case "counterfactual":
		return benchmarkCounterfactualTensors(testingObject, backend, storageDType)
	case "iv_estimate":
		return benchmarkIVEstimateTensors(testingObject, backend, storageDType)
	case "dag_markov_factorization":
		return benchmarkDAGFactorizationTensors(testingObject, backend, storageDType)
	default:
		return benchmarkCATETensors(testingObject, backend, storageDType)
	}
}

func benchmarkBackdoorTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	xCount, zCount, yCount := 4, 1024, 4
	fixture := backdoorFixtureForTest(xCount, zCount, yCount, storageDType)
	conditional, marginal, out := causalBinaryTensorsForTest(
		testingObject, backend, []int{xCount, zCount, yCount}, []int{zCount},
		[]int{xCount, yCount}, storageDType, fixture,
	)
	return []tensor.Tensor{conditional, marginal, out}
}

func benchmarkFrontdoorTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	xCount, mCount, yCount := 4, 512, 4
	fixture := frontdoorFixtureForTest(xCount, mCount, yCount, storageDType)
	mediator, outcome, marginal, out := causalTernaryTensorsForTest(
		testingObject, backend, []int{xCount, mCount}, []int{xCount, mCount, yCount},
		[]int{xCount}, []int{xCount, yCount}, storageDType, fixture,
	)
	return []tensor.Tensor{mediator, outcome, marginal, out}
}

func benchmarkDoInterveneTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	nodeCount := 128
	fixture, _ := doInterveneFixtureForTest(nodeCount, storageDType)
	matrixShape := mustShapeForTest(testingObject, []int{nodeCount, nodeCount})
	intervenedShape := mustShapeForTest(testingObject, []int{len(fixture.rightBytes) / 4})
	adjacency := uploadDTypeTensorForTest(testingObject, backend, matrixShape, storageDType, fixture.leftBytes)
	intervened := uploadDTypeTensorForTest(testingObject, backend, intervenedShape, dtype.Int32, fixture.rightBytes)
	out := emptyTensorForTest(testingObject, backend, matrixShape, storageDType)
	return []tensor.Tensor{adjacency, intervened, out}
}

func benchmarkCATETensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	fixture := cateFixtureForTest(elementCount, storageDType)
	treated, control, out := causalBinaryTensorsForTest(
		testingObject, backend, []int{elementCount}, []int{elementCount},
		[]int{elementCount}, storageDType, fixture,
	)
	return []tensor.Tensor{treated, control, out}
}

func benchmarkCounterfactualTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	fixture := counterfactualFixtureForTest(elementCount, storageDType)
	observedY, observedX, counterfactualX, slope, out := counterfactualTensorsForTest(
		testingObject, backend, elementCount, storageDType, fixture,
	)
	return []tensor.Tensor{observedY, observedX, counterfactualX, slope, out}
}

func benchmarkIVEstimateTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	fixture := ivFixtureForTest(elementCount, storageDType)
	instrument, treatment, outcome, out := causalTernaryTensorsForTest(
		testingObject, backend, []int{elementCount}, []int{elementCount},
		[]int{elementCount}, []int{1}, storageDType, fixture,
	)
	return []tensor.Tensor{instrument, treatment, outcome, out}
}

func benchmarkDAGFactorizationTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) []tensor.Tensor {
	elementCount := 8192
	fixture, parents := dagFixtureForTest(elementCount, storageDType)
	conditionals := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{elementCount}),
		storageDType, fixture.inputBytes,
	)
	parentTensor := uploadDTypeTensorForTest(
		testingObject, backend, mustShapeForTest(testingObject, []int{len(parents)}),
		dtype.Int32, int32ValuesToBytes(parents),
	)
	out := emptyTensorForTest(testingObject, backend, mustShapeForTest(testingObject, []int{1}), storageDType)
	return []tensor.Tensor{conditionals, parentTensor, out}
}

func benchmarkCausalKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	switch name {
	case "frontdoor_adjustment", "iv_estimate":
		return lookupCausalTernaryKernel(testingObject, name, storageDType)
	case "do_intervene", "dag_markov_factorization":
		return lookupCausalInt32Kernel(testingObject, name, storageDType)
	case "counterfactual":
		return lookupCausalQuaternaryKernel(testingObject, name, storageDType)
	default:
		return lookupCausalBinaryKernel(testingObject, name, storageDType)
	}
}

func causalBenchmarkBytes(name string, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))

	switch name {
	case "backdoor_adjustment":
		return int64(4*1024*4+1024+4*4) * elementBytes
	case "frontdoor_adjustment":
		return int64(4*512+4*512*4+4+4*4) * elementBytes
	case "do_intervene":
		return int64(128*128*2)*elementBytes + int64(4*4)
	case "counterfactual":
		return int64(8192*4+1) * elementBytes
	case "iv_estimate":
		return int64(8192*3+1) * elementBytes
	case "dag_markov_factorization":
		return int64(8192+1)*elementBytes + int64(8192*4)
	default:
		return int64(8192*3) * elementBytes
	}
}
