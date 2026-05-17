//go:build darwin && cgo

package metal

import (
	"context"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpucausal "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalCausalOps_CATETensor(test *testing.T) {
	Convey("Given resident Metal causal CATE inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		causalOps := causalOpsForTest(test, tensorBackend)

		Convey("It should match the scalar CATE reference at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				covariateDimensions, covariates, treatment, outcome := metalCausalCATEInputs(samples)
				expected := referenceCausalCATE(test, samples, covariateDimensions, covariates, treatment, outcome)
				output, err := causalOps.CATETensor(
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples, covariateDimensions), covariates),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples), treatment),
					uploadMetalTensorForTest(test, tensorBackend, causalShape(test, samples), outcome),
					causalShape(test, samples),
					samples,
					covariateDimensions,
				)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := output.CloneFloat64()
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalCATEMaxDiff(test, values, expected, 2e-2)
			}
		})
	})
}

func TestTensorBackend_applyCausalCATEGraph(test *testing.T) {
	Convey("Given Metal causal CATE graph execution", test, func() {
		Convey("It should keep CATE graph outputs resident", func() {
			tensorBackend := newMetalTensorBackendForTest(test)
			graph, target, expectedBytes, expected := causalCATEGraph(test)

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
			assertMetalCATEMaxDiff(test, values, expected, 2e-2)
		})
	})
}

func BenchmarkMetalCausalOps_CATETensor(benchmark *testing.B) {
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
	covariateDimensions, covariates, treatment, outcome := metalCausalCATEInputs(samples)
	covariateTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples, covariateDimensions), covariates)
	treatmentTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples), treatment)
	outcomeTensor := uploadMetalTensor(tensorBackend, causalShape(benchmark, samples), outcome)
	defer closeBenchmarkTensors(covariateTensor, treatmentTensor, outcomeTensor)

	outputShape := causalShape(benchmark, samples)
	benchmark.ResetTimer()
	for benchmark.Loop() {
		output, err := causalOps.CATETensor(
			covariateTensor,
			treatmentTensor,
			outcomeTensor,
			outputShape,
			samples,
			covariateDimensions,
		)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func metalCausalCATEInputs(samples int) (int, []float64, []float64, []float64) {
	covariateDimensions := 2
	covariates := make([]float64, samples*covariateDimensions)
	treatment := make([]float64, samples)
	outcome := make([]float64, samples)

	for sample := range samples {
		xFirst := float64(float32(-0.45 + 0.08*float64(sample%13) + 0.01*float64(sample/13)))
		xSecond := float64(float32(0.3 - 0.05*float64(sample%11) + 0.004*float64((sample*sample)%7)))
		treatmentValue := 0.0

		if sample%2 == 0 {
			treatmentValue = 1.0
		}

		base := 0.25 + 0.4*xFirst - 0.1*xSecond
		effect := 1.15 + 0.2*xFirst + 0.05*xSecond
		value := base
		if treatmentValue >= 0.5 {
			value += effect
		}

		covariates[sample*covariateDimensions] = xFirst
		covariates[sample*covariateDimensions+1] = xSecond
		treatment[sample] = treatmentValue
		outcome[sample] = float64(float32(value))
	}

	return covariateDimensions, covariates, treatment, outcome
}

func referenceCausalCATE(
	test testing.TB,
	samples int,
	covariateDimensions int,
	covariates []float64,
	treatment []float64,
	outcome []float64,
) []float64 {
	test.Helper()

	cpuState, err := cpucausal.NewCATE().Forward(
		state.NewDict().WithShape([]int{samples, covariateDimensions}).WithInputs(
			covariates,
			treatment,
			outcome,
		),
	)
	So(err, ShouldBeNil)

	return cpuState.Out
}

func causalCATEGraph(test testing.TB) (*ir.Graph, *ir.Node, int64, []float64) {
	test.Helper()

	samples := 64
	covariateDimensions, covariates, treatment, outcome := metalCausalCATEInputs(samples)
	values := [][]float64{covariates, treatment, outcome}
	names := []string{"cate_covariates", "cate_treatment", "cate_outcome"}
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
		"causal_cate",
		"causal.cate",
		causalShape(test, samples, covariateDimensions),
	)
	for _, input := range inputs {
		target.AddInput(input)
	}
	graph.AddNode(target)

	return graph, target, expectedBytes, referenceCausalCATE(
		test,
		samples,
		covariateDimensions,
		covariates,
		treatment,
		outcome,
	)
}

func assertMetalCATEMaxDiff(
	test testing.TB,
	actual []float64,
	expected []float64,
	tolerance float64,
) {
	test.Helper()

	if len(actual) != len(expected) {
		test.Fatalf("len(actual)=%d len(expected)=%d", len(actual), len(expected))
	}

	for index, expectedValue := range expected {
		actualValue := actual[index]
		if math.IsNaN(expectedValue) {
			if !math.IsNaN(actualValue) {
				test.Fatalf("index %d: expected NaN, got %v", index, actualValue)
			}

			continue
		}

		difference := math.Abs(actualValue - expectedValue)
		if difference > tolerance {
			test.Fatalf("index %d: got %v expected %v diff %v tolerance %v", index, actualValue, expectedValue, difference, tolerance)
		}
	}
}
