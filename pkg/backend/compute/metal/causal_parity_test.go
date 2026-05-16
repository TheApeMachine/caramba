//go:build darwin && cgo

package metal

import (
	"math"
	"path/filepath"
	"runtime"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	cpucausal "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/causal"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func testdataPathMetalLib(name string) string {
	_, file, _, ok := runtime.Caller(0)

	if !ok {
		return name
	}

	dir := filepath.Dir(file)

	return filepath.Join(dir, name)
}

func TestMetalCausalOps_DoCalculus_ParityWithCPU(t *testing.T) {
	Convey("Given identical Pearl do-calculus inputs", t, func() {
		dummyPath := testdataPathMetalLib("causal.metallib")
		metalOps, err := NewCausalOps(dummyPath)

		So(err, ShouldBeNil)
		defer func() {
			So(metalOps.Close(), ShouldBeNil)
		}()

		n := 3
		cov := []float64{
			1, 0.2, 0.1,
			0.2, 1, 0.3,
			0.1, 0.3, 1,
		}
		mask := []float64{0, 1, 0}
		values := []float64{0, 0.75, 0}

		cpu := cpucausal.NewDoCalculus()
		cpuState, errCPU := cpu.Forward(
			state.NewDict().WithShape([]int{n, n}).WithInputs(cov, mask, values),
		)

		metaOut, errMeta := metalOps.DoCalculus([]int{n}, cov, mask, values)

		So(errCPU, ShouldBeNil)
		So(errMeta, ShouldBeNil)
		So(len(metaOut), ShouldEqual, len(cpuState.Out))

		for idx := range cpuState.Out {
			So(math.Abs(metaOut[idx]-cpuState.Out[idx]) < 1e-4, ShouldBeTrue)
		}
	})
}

func TestMetalCausalOps_Counterfactual(t *testing.T) {
	Convey("Given identical counterfactual inputs", t, func() {
		metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

		So(err, ShouldBeNil)
		defer func() {
			So(metalOps.Close(), ShouldBeNil)
		}()

		shape := []int{3, 2}
		xObs := []float64{1, 2, 3}
		yObs := []float64{2, 5, 10}
		beta := []float64{1, 2, 3}
		xCF := []float64{4, 5}
		cpuState, errCPU := cpucausal.NewCounterfactual().Forward(
			state.NewDict().WithShape(shape).WithInputs(xObs, yObs, beta, xCF),
		)

		output, errMetal := metalOps.Counterfactual(shape, xObs, yObs, beta, xCF)

		So(errCPU, ShouldBeNil)
		So(errMetal, ShouldBeNil)
		So(len(output), ShouldEqual, len(cpuState.Out))

		for index := range output {
			So(math.Abs(output[index]-cpuState.Out[index]) < 1e-4, ShouldBeTrue)
		}
	})
}

func TestMetalCausalOps_FrontdoorAdjustment(t *testing.T) {
	Convey("Given identical frontdoor inputs", t, func() {
		metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

		So(err, ShouldBeNil)
		defer func() {
			So(metalOps.Close(), ShouldBeNil)
		}()

		shape := []int{2, 2, 1, 6}
		x := []float64{0.1, 0.2, 0.9, 1.0, 1.1, 1.2}
		mediator := []float64{0.0, 0.1, 0.8, 0.9, 1.0, 1.1}
		y := []float64{1, 1.5, 3, 3.5, 4, 4.5}
		cpuState, errCPU := cpucausal.NewFrontdoorAdjustment().Forward(
			state.NewDict().WithShape(shape).WithInputs(x, mediator, y),
		)

		output, errMetal := metalOps.FrontdoorAdjustment(shape, x, mediator, y)

		So(errCPU, ShouldBeNil)
		So(errMetal, ShouldBeNil)
		So(len(output), ShouldEqual, len(cpuState.Out))

		for index := range output {
			So(math.Abs(output[index]-cpuState.Out[index]) < 1e-4, ShouldBeTrue)
		}
	})
}

func TestMetalCausalOps_FrontdoorAdjustmentContractSizes(test *testing.T) {
	Convey("Given Metal frontdoor adjustment", test, func() {
		metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

		So(err, ShouldBeNil)
		defer func() {
			So(metalOps.Close(), ShouldBeNil)
		}()

		Convey("It should match CPU equal-frequency binning at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				treatmentBins, mediatorBins, treatment, mediator, outcome := metalCausalFrontdoorInputs(samples)
				shape := []int{treatmentBins, mediatorBins, 1, samples}
				cpuState, cpuErr := cpucausal.NewFrontdoorAdjustment().Forward(
					state.NewDict().WithShape(shape).WithInputs(treatment, mediator, outcome),
				)
				metalOutput, metalErr := metalOps.FrontdoorAdjustment(
					shape,
					treatment,
					mediator,
					outcome,
				)

				So(cpuErr, ShouldBeNil)
				So(metalErr, ShouldBeNil)
				assertMetalMaxDiff(metalOutput, cpuState.Out, 1e-4)
			}
		})
	})
}

func TestMetalCausalOps_DAGMarkovFactorization(test *testing.T) {
	Convey("Given Metal DAG Markov factorization", test, func() {
		metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

		So(err, ShouldBeNil)
		defer func() {
			So(metalOps.Close(), ShouldBeNil)
		}()

		Convey("It should match CPU state-dict execution at contract sizes", func() {
			for _, samples := range metalContractSizes() {
				nodeCount, observations, adjacency := metalCausalDAGInputs(samples)
				cpuState, cpuErr := cpucausal.NewDAGMarkovFactorization().Forward(
					state.NewDict().WithShape([]int{nodeCount, samples}).WithInputs(
						observations,
						adjacency,
					),
				)
				metalOutput, metalErr := metalOps.DAGMarkovFactorization(
					[]int{nodeCount, samples},
					observations,
					adjacency,
				)

				So(cpuErr, ShouldBeNil)
				So(metalErr, ShouldBeNil)
				assertMetalMaxDiff(metalOutput, cpuState.Out, 2e-2)
			}
		})
	})
}

func BenchmarkMetalCausalOps_Counterfactual(b *testing.B) {
	metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = metalOps.Close()
	}()

	shape := []int{3, 2}
	xObs := []float64{1, 2, 3}
	yObs := []float64{2, 5, 10}
	beta := []float64{1, 2, 3}
	xCF := []float64{4, 5}

	b.ResetTimer()

	for b.Loop() {
		_, _ = metalOps.Counterfactual(shape, xObs, yObs, beta, xCF)
	}
}

func BenchmarkMetalCausalOps_DAGMarkovFactorization(benchmark *testing.B) {
	metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = metalOps.Close()
	}()

	samples := 8192
	nodeCount, observations, adjacency := metalCausalDAGInputs(samples)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := metalOps.DAGMarkovFactorization(
			[]int{nodeCount, samples},
			observations,
			adjacency,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkMetalCausalOps_FrontdoorAdjustment(benchmark *testing.B) {
	metalOps, err := NewCausalOps(testdataPathMetalLib("causal.metallib"))

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = metalOps.Close()
	}()

	samples := 8192
	treatmentBins, mediatorBins, treatment, mediator, outcome := metalCausalFrontdoorInputs(samples)
	shape := []int{treatmentBins, mediatorBins, 1, samples}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := metalOps.FrontdoorAdjustment(
			shape,
			treatment,
			mediator,
			outcome,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func metalCausalDAGInputs(samples int) (int, []float64, []float64) {
	if samples == 1 {
		return 1, []float64{0.25}, []float64{0}
	}

	nodeCount := 3
	observations := make([]float64, samples*nodeCount)
	adjacency := []float64{
		0, 0, 0,
		1, 0, 0,
		1, 0, 0,
	}

	for sampleIndex := range samples {
		root := -0.4 +
			0.09*float64(sampleIndex%17-8) +
			0.02*float64((sampleIndex/17)%5)
		middle := 0.2 + 0.55*root + 0.07*float64(sampleIndex%3-1)
		leaf := -0.1 - 0.35*root + 0.05*float64(sampleIndex%7-3)
		offset := sampleIndex * nodeCount
		observations[offset] = root
		observations[offset+1] = middle
		observations[offset+2] = leaf
	}

	return nodeCount, observations, adjacency
}

func metalCausalFrontdoorInputs(samples int) (int, int, []float64, []float64, []float64) {
	if samples == 1 {
		return 1, 1, []float64{0.25}, []float64{0.35}, []float64{0.5}
	}

	treatmentBins := 3
	mediatorBins := 2
	treatment := make([]float64, samples)
	mediator := make([]float64, samples)
	outcome := make([]float64, samples)

	for sampleIndex := range samples {
		xValue := -0.5 +
			0.12*float64(sampleIndex%11) +
			0.03*float64((sampleIndex/11)%3)
		mediatorValue := 0.2 + 0.4*xValue + 0.05*float64(sampleIndex%5-2)
		outcomeValue := 1.0 + 0.7*xValue - 0.3*mediatorValue + 0.02*float64(sampleIndex%3-1)

		treatment[sampleIndex] = xValue
		mediator[sampleIndex] = mediatorValue
		outcome[sampleIndex] = outcomeValue
	}

	return treatmentBins, mediatorBins, treatment, mediator, outcome
}
