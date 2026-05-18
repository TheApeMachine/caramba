//go:build darwin && cgo

package metal

import (
	"context"
	"slices"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cputrain "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/train"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestTensorBackend_applyOptimizerGraph(test *testing.T) {
	Convey("Given Metal optimizer graph execution", test, func() {
		Convey("It should match CPU optimizer steps and keep outputs resident", func() {
			for _, operation := range optimizerTensorOperations() {
				for _, elementCount := range metalContractSizes() {
					params, grads := optimizerTensorValues(elementCount)
					expected, err := referenceOptimizerTensor(operation, params, grads)
					So(err, ShouldBeNil)

					tensorBackend := newMetalTensorBackendForTest(test)
					graph, target := optimizerTensorGraph(test, operation, params, grads)
					before := tensorBackend.runtime.Metrics()
					results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{target})
					after := tensorBackend.runtime.Metrics()
					So(err, ShouldBeNil)
					So(results, ShouldHaveLength, 1)
					So(after.TransferBytes-before.TransferBytes, ShouldEqual, int64(elementCount*8))

					output := results[target.ID()]
					So(output.Location(), ShouldEqual, computetensor.Metal)
					values, err := tensorFloat64Values(output)
					So(err, ShouldBeNil)
					assertMetalMaxDiff(values, expected, 8e-4)
					So(output.Close(), ShouldBeNil)
				}
			}
		})

		Convey("It should preserve optimizer state across repeated graph steps", func() {
			params, grads := optimizerTensorValues(64)
			tensorBackend := newMetalTensorBackendForTest(test)
			paramTensor := uploadMetalTensor(tensorBackend, causalShape(test, len(params)), params)
			gradTensor := uploadMetalTensor(tensorBackend, causalShape(test, len(grads)), grads)
			defer closeBenchmarkTensors(paramTensor, gradTensor)

			node := executor.NodeSpec{
				ID:    "stateful_adam",
				Op:    "train.optimizer.adam",
				Shape: []int{len(params)},
			}

			referenceStep := cputrain.NewAdamStep(1e-3, 0.9, 0.999, 1e-8, 0)
			firstReference, err := referenceStep.Forward(
				state.NewDict().WithParams(params).WithGrads(grads),
			)
			So(err, ShouldBeNil)
			firstExpected := slices.Clone(firstReference.Out)
			secondReference, err := referenceStep.Forward(
				state.NewDict().WithParams(firstExpected).WithGrads(grads),
			)
			So(err, ShouldBeNil)
			secondExpected := slices.Clone(secondReference.Out)

			firstOutput, err := tensorBackend.Apply(
				context.Background(),
				node,
				[]computetensor.Tensor{paramTensor, gradTensor},
			)
			So(err, ShouldBeNil)
			defer func() {
				So(firstOutput.Close(), ShouldBeNil)
			}()

			firstValues, err := tensorFloat64Values(firstOutput)
			So(err, ShouldBeNil)
			assertMetalMaxDiff(firstValues, firstExpected, 8e-4)

			secondOutput, err := tensorBackend.Apply(
				context.Background(),
				node,
				[]computetensor.Tensor{firstOutput, gradTensor},
			)
			So(err, ShouldBeNil)
			defer func() {
				So(secondOutput.Close(), ShouldBeNil)
			}()

			secondValues, err := tensorFloat64Values(secondOutput)
			So(err, ShouldBeNil)
			assertMetalMaxDiff(secondValues, secondExpected, 8e-4)
		})
	})
}

func BenchmarkMetalOptimizerTensor_Adam(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.adam")
}

func BenchmarkMetalOptimizerTensor_AdamW(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.adamw")
}

func BenchmarkMetalOptimizerTensor_AdaMax(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.adamax")
}

func BenchmarkMetalOptimizerTensor_SGD(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.sgd")
}

func BenchmarkMetalOptimizerTensor_Lion(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.lion")
}

func BenchmarkMetalOptimizerTensor_RMSProp(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.rmsprop")
}

func BenchmarkMetalOptimizerTensor_Hebbian(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.hebbian")
}

func BenchmarkMetalOptimizerTensor_LARS(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.lars")
}

func BenchmarkMetalOptimizerTensor_LAMB(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.lamb")
}

func BenchmarkMetalOptimizerTensor_AdaGrad(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.adagrad")
}

func BenchmarkMetalOptimizerTensor_AdaDelta(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.adadelta")
}

func BenchmarkMetalOptimizerTensor_LBFGS(benchmark *testing.B) {
	benchmarkOptimizerTensor(benchmark, "train.optimizer.lbfgs")
}

func optimizerTensorOperations() []string {
	return []string{
		"train.optimizer.adam",
		"train.optimizer.adamw",
		"train.optimizer.adamax",
		"train.optimizer.sgd",
		"train.optimizer.lion",
		"train.optimizer.rmsprop",
		"train.optimizer.hebbian",
		"train.optimizer.lars",
		"train.optimizer.lamb",
		"train.optimizer.adagrad",
		"train.optimizer.adadelta",
		"train.optimizer.lbfgs",
	}
}

func optimizerTensorGraph(
	test testing.TB,
	operation string,
	params []float64,
	grads []float64,
) (*ir.Graph, *ir.Node) {
	test.Helper()

	graph := ir.NewGraph()
	paramNode := ir.NewNode("optimizer_params_"+operation, ir.OpInput, causalShape(test, len(params)))
	gradNode := ir.NewNode("optimizer_grads_"+operation, ir.OpInput, causalShape(test, len(grads)))
	paramNode.SetMetadata("values", params)
	gradNode.SetMetadata("values", grads)
	target := ir.NewNode("optimizer_"+operation, ir.OpType(operation), causalShape(test, len(params)))
	target.AddInput(paramNode)
	target.AddInput(gradNode)
	graph.AddNode(paramNode)
	graph.AddNode(gradNode)
	graph.AddNode(target)

	return graph, target
}

func referenceOptimizerTensor(
	operation string,
	params []float64,
	grads []float64,
) ([]float64, error) {
	optimizerStep := optimizerTensorReferenceStep(operation)
	updated, err := optimizerStep.Forward(state.NewDict().WithParams(params).WithGrads(grads))
	if err != nil {
		return nil, err
	}

	return updated.Out, nil
}

func optimizerTensorReferenceStep(operation string) *cputrain.OptimizerStep {
	switch operation {
	case "train.optimizer.adam":
		return cputrain.NewAdamStep(1e-3, 0.9, 0.999, 1e-8, 0)
	case "train.optimizer.adamw":
		return cputrain.NewAdamWStep(1e-3, 0.9, 0.999, 1e-8, 0)
	case "train.optimizer.adamax":
		return cputrain.NewAdaMaxStep(2e-3, 0.9, 0.999, 1e-8)
	case "train.optimizer.sgd":
		return cputrain.NewSGDStep(1e-3, 0, 0, false)
	case "train.optimizer.lion":
		return cputrain.NewLionStep(1e-4, 0.9, 0.99, 0)
	case "train.optimizer.rmsprop":
		return cputrain.NewRMSPropStep(1e-2, 0.99, 1e-8, 0, 0)
	case "train.optimizer.hebbian":
		return cputrain.NewHebbianStep(1e-3, 0)
	case "train.optimizer.lars":
		return cputrain.NewLARSStep(1e-2, 1e-3, 0.9, 0, 1e-8)
	case "train.optimizer.lamb":
		return cputrain.NewLAMBStep(1e-3, 0.9, 0.999, 1e-6, 0)
	case "train.optimizer.adagrad":
		return cputrain.NewAdaGradStep(1e-2, 1e-10, 0, 0)
	case "train.optimizer.adadelta":
		return cputrain.NewAdaDeltaStep(0.9, 1e-6, 0)
	case "train.optimizer.lbfgs":
		return cputrain.NewLBFGSStep(1.0, 10, false, 1e-4)
	}

	return cputrain.NewSGDStep(1e-3, 0, 0, false)
}

func benchmarkOptimizerTensor(benchmark *testing.B, operation string) {
	benchmark.ReportAllocs()
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	params, grads := optimizerTensorValues(8192)
	paramTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, len(params)), params)
	gradTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, len(grads)), grads)
	defer closeBenchmarkTensors(paramTensor, gradTensor)

	node := executor.NodeSpec{
		ID:    "optimizer_benchmark_" + operation,
		Op:    ir.OpType(operation),
		Shape: []int{len(params)},
	}

	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := tensorBackend.Apply(
			context.Background(),
			node,
			[]computetensor.Tensor{paramTensor, gradTensor},
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func optimizerTensorValues(elementCount int) ([]float64, []float64) {
	params := make([]float64, elementCount)
	grads := make([]float64, elementCount)

	for index := range elementCount {
		sign := 1.0
		if index%2 != 0 {
			sign = -1.0
		}

		params[index] = float64(float32(sign*(0.17+0.031*float64(index%11)) + 0.0007*float64(index/11)))
		grads[index] = float64(float32(-sign*(0.11+0.023*float64(index%13)) + 0.0004*float64(index/13)))
	}

	return params, grads
}
