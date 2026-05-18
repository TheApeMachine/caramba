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

func TestMetalCausalOps_BackdoorAdjustmentTensor(test *testing.T) {
	Convey("Given resident Metal causal backdoor inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the scalar backdoor reference at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				outcomeDimensions, treatmentDimensions, confounderDimensions, outcome, treatment, confounder := metalCausalBackdoorInputs(samples)
				expected := referenceCausalBackdoor(
					test,
					outcomeDimensions,
					treatmentDimensions,
					confounderDimensions,
					samples,
					outcome,
					treatment,
					confounder,
				)
				output, err := causalOps.BackdoorAdjustmentTensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, outcomeDimensions), outcome),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, treatmentDimensions), treatment),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, confounderDimensions), confounder),
					causalShape(test, outcomeDimensions),
					samples,
					outcomeDimensions,
					treatmentDimensions,
					confounderDimensions,
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

func TestTensorBackend_applyCausalBackdoorGraph(test *testing.T) {
	Convey("Given Metal causal backdoor graph execution", test, func() {
		Convey("It should keep backdoor graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalBackdoorGraph(test)

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

func BenchmarkMetalCausalOps_BackdoorAdjustmentTensor(benchmark *testing.B) {
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
	outcomeDimensions, treatmentDimensions, confounderDimensions, outcome, treatment, confounder := metalCausalBackdoorInputs(samples)
	outcomeTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, outcomeDimensions), outcome)
	treatmentTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, treatmentDimensions), treatment)
	confounderTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, confounderDimensions), confounder)
	defer closeBenchmarkTensors(outcomeTensor, treatmentTensor, confounderTensor)

	outputShape := causalShape(benchmark, outcomeDimensions)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.BackdoorAdjustmentTensor(
			outcomeTensor,
			treatmentTensor,
			confounderTensor,
			outputShape,
			samples,
			outcomeDimensions,
			treatmentDimensions,
			confounderDimensions,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func metalCausalBackdoorInputs(
	samples int,
) (int, int, int, []float64, []float64, []float64) {
	outcomeDimensions, treatmentDimensions, confounderDimensions := 2, 2, 1
	outcome := make([]float64, samples*outcomeDimensions)
	treatment := make([]float64, samples*treatmentDimensions)
	confounder := make([]float64, samples*confounderDimensions)

	if samples == 1 {
		return outcomeDimensions, treatmentDimensions, confounderDimensions, outcome, treatment, confounder
	}

	for sample := range samples {
		xFirst := float64(float32(-0.35 + 0.11*float64(sample%11) + 0.01*float64(sample/11)))
		xSecond := float64(float32(0.41 - 0.07*float64(sample%7) + 0.005*float64((sample*sample)%5)))
		zValue := float64(float32(-0.2 + 0.09*float64((sample*3)%13)))

		treatment[sample*treatmentDimensions] = xFirst
		treatment[sample*treatmentDimensions+1] = xSecond
		confounder[sample] = zValue
		outcome[sample*outcomeDimensions] = float64(float32(0.3 + 1.2*xFirst - 0.4*xSecond + 0.25*zValue))
		outcome[sample*outcomeDimensions+1] = float64(float32(-0.1 + 0.5*xFirst + 0.9*xSecond - 0.15*zValue))
	}

	return outcomeDimensions, treatmentDimensions, confounderDimensions, outcome, treatment, confounder
}

func referenceCausalBackdoor(
	test testing.TB,
	outcomeDimensions int,
	treatmentDimensions int,
	confounderDimensions int,
	samples int,
	outcome []float64,
	treatment []float64,
	confounder []float64,
) []float64 {
	test.Helper()

	cpuState, err := cpucausal.NewBackdoorAdjustment().Forward(
		state.NewDict().WithShape([]int{
			outcomeDimensions,
			treatmentDimensions,
			confounderDimensions,
			samples,
		}).WithInputs(
			outcome,
			treatment,
			confounder,
		),
	)
	So(err, ShouldBeNil)

	return cpuState.Out
}

func causalBackdoorGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	samples := 64
	outcomeDimensions, treatmentDimensions, confounderDimensions, outcome, treatment, confounder := metalCausalBackdoorInputs(samples)
	values := [][]float64{outcome, treatment, confounder}
	names := []string{"backdoor_outcome", "backdoor_treatment", "backdoor_confounder"}
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
		"causal_backdoor",
		"causal.backdoor_adjustment",
		causalShape(test, outcomeDimensions, treatmentDimensions, confounderDimensions, samples),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalBackdoor(
		test,
		outcomeDimensions,
		treatmentDimensions,
		confounderDimensions,
		samples,
		outcome,
		treatment,
		confounder,
	)
}
