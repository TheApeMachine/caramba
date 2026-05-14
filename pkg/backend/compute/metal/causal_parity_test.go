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
