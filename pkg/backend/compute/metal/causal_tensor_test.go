//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpucausal "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalCausalOps_CounterfactualTensor(test *testing.T) {
	Convey("Given resident Metal causal counterfactual inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar counterfactual reference", func() {
			for _, observedCount := range metalContractSizes() {
				counterfactualCount := 3
				observedX, observedY, beta, counterfactualX := causalCounterfactualInputs(
					observedCount,
					counterfactualCount,
				)
				output, err := causalOps.CounterfactualTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, observedCount), observedX),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, observedCount), observedY),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, observedCount), beta),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, counterfactualCount), counterfactualX),
					causalShape(test, observedCount, counterfactualCount),
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(
					values,
					referenceCausalCounterfactual(observedX, observedY, beta, counterfactualX),
					1e-6,
				)
			}
		})
	})
}

func TestTensorBackend_applyCausalCounterfactualGraph(test *testing.T) {
	Convey("Given Metal causal counterfactual graph execution", test, func() {
		Convey("It should keep causal graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalCounterfactualGraph(test)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{target})
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)

			output := results[target.ID()]
			So(output.Location(), ShouldEqual, computetensor.Metal)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalMaxDiff(values, expected, 1e-6)
		})
	})
}

func TestMetalCausalOps_FrontdoorAdjustmentTensor(test *testing.T) {
	Convey("Given resident Metal causal frontdoor inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the scalar frontdoor reference at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				treatmentBins, mediatorBins, treatment, mediator, outcome := metalCausalFrontdoorInputs(samples)
				expected := referenceCausalFrontdoor(test, treatmentBins, mediatorBins, samples, treatment, mediator, outcome)
				output, err := causalOps.FrontdoorAdjustmentTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, len(treatment)), treatment),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, len(mediator)), mediator),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, len(outcome)), outcome),
					causalShape(test, treatmentBins),
					samples,
					treatmentBins,
					mediatorBins,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, expected, 1e-4)
			}
		})
	})
}

func TestTensorBackend_applyCausalFrontdoorGraph(test *testing.T) {
	Convey("Given Metal causal frontdoor graph execution", test, func() {
		Convey("It should keep frontdoor graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalFrontdoorGraph(test)

			before := tensorBackend.runtime.Metrics()
			results, err := NewRunnerWithBackend(tensorBackend).Execute(context.Background(), graph, []*ir.Node{target})
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			So(results, ShouldHaveLength, 1)
			So(after.TransferBytes-before.TransferBytes, ShouldEqual, expectedBytes)

			output := results[target.ID()]
			So(output.Location(), ShouldEqual, computetensor.Metal)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalMaxDiff(values, expected, 1e-4)
		})
	})
}

func BenchmarkMetalCausalOps_CounterfactualTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	causalOps, err := tensorBackend.causal()
	if err != nil {
		benchmark.Fatal(err)
	}

	observedCount, counterfactualCount := 8192, 8
	observedX, observedY, beta, counterfactualX := causalCounterfactualInputs(
		observedCount,
		counterfactualCount,
	)
	observedXTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, observedCount), observedX)
	observedYTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, observedCount), observedY)
	betaTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, observedCount), beta)
	counterfactualTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, counterfactualCount), counterfactualX)
	defer closeBenchmarkTensors(observedXTensor, observedYTensor, betaTensor, counterfactualTensor)

	outputShape := causalShape(benchmark, observedCount, counterfactualCount)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.CounterfactualTensor(
			observedXTensor,
			observedYTensor,
			betaTensor,
			counterfactualTensor,
			outputShape,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func BenchmarkMetalCausalOps_FrontdoorAdjustmentTensor(benchmark *testing.B) {
	benchmark.ReportAllocs()
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	causalOps, err := tensorBackend.causal()
	if err != nil {
		benchmark.Fatal(err)
	}

	samples := 8192
	treatmentBins, mediatorBins, treatment, mediator, outcome := metalCausalFrontdoorInputs(samples)
	treatmentTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples), treatment)
	mediatorTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples), mediator)
	outcomeTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples), outcome)
	defer closeBenchmarkTensors(treatmentTensor, mediatorTensor, outcomeTensor)

	outputShape := causalShape(benchmark, treatmentBins)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.FrontdoorAdjustmentTensor(
			treatmentTensor,
			mediatorTensor,
			outcomeTensor,
			outputShape,
			samples,
			treatmentBins,
			mediatorBins,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func causalOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalCausalOps {
	test.Helper()

	causalOps, err := tensorBackend.causal()
	So(err, ShouldBeNil)

	return causalOps
}

func causalShape(test testing.TB, dimensions ...int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape(dimensions)
	if err != nil {
		test.Fatalf("causal shape: %v", err)
	}

	return shape
}

func causalCounterfactualInputs(
	observedCount int,
	counterfactualCount int,
) ([]float64, []float64, []float64, []float64) {
	observedX := make([]float64, observedCount)
	observedY := make([]float64, observedCount)
	beta := make([]float64, observedCount)
	counterfactualX := make([]float64, counterfactualCount)

	for index := range observedCount {
		observedX[index] = float64(float32(0.1 + 0.013*float64(index%17-8)))
		observedY[index] = float64(float32(0.2 + 0.019*float64(index%19-9)))
		beta[index] = float64(float32(0.35 + 0.007*float64(index%11-5)))
	}

	for index := range counterfactualCount {
		counterfactualX[index] = float64(float32(-0.2 + 0.17*float64(index)))
	}

	return observedX, observedY, beta, counterfactualX
}

func referenceCausalCounterfactual(
	observedX []float64,
	observedY []float64,
	beta []float64,
	counterfactualX []float64,
) []float64 {
	values := make([]float64, len(observedX)*len(counterfactualX))

	for observedIndex := range observedX {
		for counterfactualIndex, value := range counterfactualX {
			offset := observedIndex*len(counterfactualX) + counterfactualIndex
			values[offset] = float64(
				float32(beta[observedIndex])*float32(value) +
					float32(observedY[observedIndex]) -
					float32(beta[observedIndex])*float32(observedX[observedIndex]),
			)
		}
	}

	return values
}

func causalCounterfactualGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	observedCount, counterfactualCount := 64, 4
	observedX, observedY, beta, counterfactualX := causalCounterfactualInputs(
		observedCount,
		counterfactualCount,
	)
	values := [][]float64{observedX, observedY, beta, counterfactualX}
	names := []string{"causal_x_obs", "causal_y_obs", "causal_beta", "causal_x_cf"}
	graph := ir.NewGraph()
	inputs := make([]*ir.Node, len(values))
	expectedBytes := int64(0)

	for index, value := range values {
		node := ir.NewNode(names[index], ir.OpInput, causalShape(test, len(value)))
		node.SetMetadata("values", value)
		graph.AddNode(node)
		inputs[index] = node
		expectedBytes += int64(len(value) * 4)
	}

	target := ir.NewNode(
		"causal_counterfactual",
		"causal.counterfactual",
		causalShape(test, observedCount, counterfactualCount),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalCounterfactual(
		observedX,
		observedY,
		beta,
		counterfactualX,
	)
}

func referenceCausalFrontdoor(
	test testing.TB,
	treatmentBins int,
	mediatorBins int,
	samples int,
	treatment []float64,
	mediator []float64,
	outcome []float64,
) []float64 {
	test.Helper()

	cpuState, err := cpucausal.NewFrontdoorAdjustment().Forward(
		state.NewDict().WithShape([]int{treatmentBins, mediatorBins, 1, samples}).WithInputs(
			treatment,
			mediator,
			outcome,
		),
	)
	So(err, ShouldBeNil)

	return cpuState.Out
}

func causalFrontdoorGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	samples := 64
	treatmentBins, mediatorBins, treatment, mediator, outcome := metalCausalFrontdoorInputs(samples)
	values := [][]float64{treatment, mediator, outcome}
	names := []string{"frontdoor_treatment", "frontdoor_mediator", "frontdoor_outcome"}
	graph := ir.NewGraph()
	inputs := make([]*ir.Node, len(values))
	expectedBytes := int64(0)

	for index, value := range values {
		node := ir.NewNode(names[index], ir.OpInput, causalShape(test, len(value)))
		node.SetMetadata("values", value)
		graph.AddNode(node)
		inputs[index] = node
		expectedBytes += int64(len(value) * 4)
	}

	target := ir.NewNode(
		"causal_frontdoor",
		"causal.frontdoor_adjustment",
		causalShape(test, treatmentBins, mediatorBins, 1, samples),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalFrontdoor(
		test,
		treatmentBins,
		mediatorBins,
		samples,
		treatment,
		mediator,
		outcome,
	)
}
