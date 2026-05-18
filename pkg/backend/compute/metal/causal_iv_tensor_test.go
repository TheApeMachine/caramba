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

func TestMetalCausalOps_IVEstimateTensor(test *testing.T) {
	Convey("Given resident Metal causal IV inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the scalar IV reference at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				instrumentDimensions, treatmentDimensions, outcomeDimensions, instrument, treatment, outcome := metalCausalIVInputs(samples)
				expected := referenceCausalIV(
					test,
					samples,
					instrumentDimensions,
					treatmentDimensions,
					outcomeDimensions,
					instrument,
					treatment,
					outcome,
				)
				output, err := causalOps.IVEstimateTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, instrumentDimensions), instrument),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, treatmentDimensions), treatment),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, outcomeDimensions), outcome),
					causalShape(test, treatmentDimensions, outcomeDimensions),
					samples,
					instrumentDimensions,
					treatmentDimensions,
					outcomeDimensions,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, expected, 3e-3)
			}
		})
	})
}

func TestTensorBackend_applyCausalIVEstimateGraph(test *testing.T) {
	Convey("Given Metal causal IV graph execution", test, func() {
		Convey("It should keep IV graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalIVGraph(test)

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

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertMetalMaxDiff(values, expected, 3e-3)
		})
	})
}

func BenchmarkMetalCausalOps_IVEstimateTensor(benchmark *testing.B) {
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
	instrumentDimensions, treatmentDimensions, outcomeDimensions, instrument, treatment, outcome := metalCausalIVInputs(samples)
	instrumentTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, instrumentDimensions), instrument)
	treatmentTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, treatmentDimensions), treatment)
	outcomeTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, outcomeDimensions), outcome)
	defer closeBenchmarkTensors(instrumentTensor, treatmentTensor, outcomeTensor)

	outputShape := causalShape(benchmark, treatmentDimensions, outcomeDimensions)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.IVEstimateTensor(
			instrumentTensor,
			treatmentTensor,
			outcomeTensor,
			outputShape,
			samples,
			instrumentDimensions,
			treatmentDimensions,
			outcomeDimensions,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func metalCausalIVInputs(samples int) (int, int, int, []float64, []float64, []float64) {
	instrumentDimensions, treatmentDimensions, outcomeDimensions := 1, 1, 1
	instrument := make([]float64, samples*instrumentDimensions)
	treatment := make([]float64, samples*treatmentDimensions)
	outcome := make([]float64, samples*outcomeDimensions)

	if samples == 1 {
		return instrumentDimensions, treatmentDimensions, outcomeDimensions, instrument, treatment, outcome
	}

	for sample := range samples {
		instrumentValue := float64(float32(-0.35 + 0.07*float64(sample%17) + 0.002*float64(sample)))
		treatmentValue := float64(float32(0.2 + 1.7*instrumentValue))
		outcomeValue := float64(float32(-0.1 + 2.1*treatmentValue))

		instrument[sample] = instrumentValue
		treatment[sample] = treatmentValue
		outcome[sample] = outcomeValue
	}

	return instrumentDimensions, treatmentDimensions, outcomeDimensions, instrument, treatment, outcome
}

func referenceCausalIV(
	test testing.TB,
	samples int,
	instrumentDimensions int,
	treatmentDimensions int,
	outcomeDimensions int,
	instrument []float64,
	treatment []float64,
	outcome []float64,
) []float64 {
	test.Helper()

	cpuState, err := cpucausal.NewIVEstimate().Forward(
		state.NewDict().WithShape([]int{
			samples,
			instrumentDimensions,
			treatmentDimensions,
			outcomeDimensions,
		}).WithInputs(
			instrument,
			treatment,
			outcome,
		),
	)
	So(err, ShouldBeNil)

	return cpuState.Out
}

func causalIVGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	samples := 64
	instrumentDimensions, treatmentDimensions, outcomeDimensions, instrument, treatment, outcome := metalCausalIVInputs(samples)
	values := [][]float64{instrument, treatment, outcome}
	names := []string{"iv_instrument", "iv_treatment", "iv_outcome"}
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
		"causal_iv",
		"causal.iv_estimate",
		causalShape(test, samples, instrumentDimensions, treatmentDimensions, outcomeDimensions),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalIV(
		test,
		samples,
		instrumentDimensions,
		treatmentDimensions,
		outcomeDimensions,
		instrument,
		treatment,
		outcome,
	)
}
