//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpulbfgs "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lbfgs"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLBFGS_Step(test *testing.T) {
	Convey("Given a Metal L-BFGS optimizer", test, func() {
		metalOptimizer, err := NewOptimizerRegistry().LBFGS(state.NewDict())
		cpuOptimizer := cpulbfgs.NewLBFGS()

		So(err, ShouldBeNil)

		Convey("It should match the CPU two-loop update after history is populated", func() {
			cpuState := lbfgsState(
				[]float64{1.0, -0.5, 0.25, 1.5},
				[]float64{0.30, -0.10, 0.20, -0.40},
			)
			metalState := lbfgsState(
				[]float64{1.0, -0.5, 0.25, 1.5},
				[]float64{0.30, -0.10, 0.20, -0.40},
			)

			_, err = cpuOptimizer.Step(cpuState)
			So(err, ShouldBeNil)
			_, err = metalOptimizer.Step(metalState)
			So(err, ShouldBeNil)

			cpuState.WithParams([]float64{0.92, -0.42, 0.18, 1.36}).
				WithGrads([]float64{0.18, -0.04, 0.12, -0.22})
			metalState.WithParams([]float64{0.92, -0.42, 0.18, 1.36}).
				WithGrads([]float64{0.18, -0.04, 0.12, -0.22})

			cpuUpdated, err := cpuOptimizer.Step(cpuState)
			So(err, ShouldBeNil)
			metalUpdated, err := metalOptimizer.Step(metalState)
			So(err, ShouldBeNil)

			So(metalUpdated.Out, ShouldHaveLength, len(cpuUpdated.Out))
			for index := range cpuUpdated.Out {
				So(metalUpdated.Out[index], ShouldAlmostEqual, cpuUpdated.Out[index], 1e-4)
			}
			So(metalState.Head, ShouldEqual, cpuState.Head)
			So(metalState.Count, ShouldEqual, cpuState.Count)
		})
	})
}

func BenchmarkLBFGS_Step(benchmark *testing.B) {
	optimizer, err := NewOptimizerRegistry().LBFGS(state.NewDict())
	if err != nil {
		benchmark.Fatal(err)
	}

	stateDict := lbfgsState(
		[]float64{1.0, -0.5, 0.25, 1.5, -1.25, 0.75, -0.2, 0.6},
		[]float64{0.30, -0.10, 0.20, -0.40, 0.14, -0.08, 0.05, -0.12},
	)

	for benchmark.Loop() {
		_, _ = optimizer.Step(stateDict)
		stateDict.WithParams(stateDict.Out).
			WithGrads([]float64{0.18, -0.04, 0.12, -0.22, 0.11, -0.05, 0.03, -0.09})
	}
}

func lbfgsState(params, grads []float64) *state.Dict {
	return state.NewDict().
		WithLR(0.4).
		WithHistSize(4).
		WithLineSearch(false).
		WithC1(1e-4).
		WithParams(params).
		WithGrads(grads)
}
