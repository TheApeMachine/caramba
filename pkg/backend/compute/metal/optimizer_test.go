//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuadagrad "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/adagrad"
	cpuadam "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/adam"
	cpuhebbian "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/hebbian"
	cpulars "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lars"
	cpulbfgs "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lbfgs"
	cpulion "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lion"
	cpurmsprop "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/rmsprop"
	cpusgd "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/sgd"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestOptimizerContract_Step(test *testing.T) {
	Convey("Given Metal optimizers for the full train optimizer contract", test, func() {
		for _, optimizerCase := range metalOptimizerCases() {
			optimizerCase := optimizerCase

			Convey("It should match CPU state-dict execution for "+optimizerCase.name, func() {
				for _, length := range []int{1, 7, 64, 1024, 8192} {
					cpuState := optimizerState(length)
					metalState := optimizerState(length)
					cpuOptimizer := optimizerCase.cpu()
					metalOptimizer, err := optimizerCase.metal(NewOptimizerRegistry())

					So(err, ShouldBeNil)

					cpuUpdated, err := cpuOptimizer.Step(cpuState)
					So(err, ShouldBeNil)
					metalUpdated, err := metalOptimizer.Step(metalState)
					So(err, ShouldBeNil)

					assertMetalSlice(
						fmt.Sprintf("%s/out/%d", optimizerCase.name, length),
						metalUpdated.Out,
						cpuUpdated.Out,
						optimizerCase.tolerance,
					)

					for _, field := range optimizerCase.fields {
						assertMetalSlice(
							fmt.Sprintf("%s/%s/%d", optimizerCase.name, field, length),
							stateField(metalUpdated, field),
							stateField(cpuUpdated, field),
							optimizerCase.tolerance,
						)
					}

					So(metalUpdated.Step, ShouldEqual, cpuUpdated.Step)
				}
			})
		}
	})
}

func TestLBFGS_Step(test *testing.T) {
	Convey("Given a Metal L-BFGS optimizer", test, func() {
		metalOptimizer, err := NewOptimizerRegistry().LBFGS(state.NewDict())
		cpuOptimizer := cpulbfgs.NewLBFGS()

		So(err, ShouldBeNil)

		Convey("It should match the CPU two-loop update after history is populated", func() {
			for _, length := range []int{1, 7, 64, 1024, 8192} {
				initialParams := pattern(length, 0.41, 0.017)
				initialGrads := pattern(length, -0.23, 0.011)
				nextParams := lbfgsShift(initialParams, initialGrads, -0.08)
				nextGrads := lbfgsShift(initialGrads, initialGrads, -0.25)
				cpuState := lbfgsState(initialParams, initialGrads)
				metalState := lbfgsState(
					append([]float64(nil), initialParams...),
					append([]float64(nil), initialGrads...),
				)

				_, err = cpuOptimizer.Step(cpuState)
				So(err, ShouldBeNil)
				_, err = metalOptimizer.Step(metalState)
				So(err, ShouldBeNil)

				cpuState.WithParams(nextParams).WithGrads(nextGrads)
				metalState.WithParams(
					append([]float64(nil), nextParams...),
				).WithGrads(
					append([]float64(nil), nextGrads...),
				)

				cpuUpdated, err := cpuOptimizer.Step(cpuState)
				So(err, ShouldBeNil)
				metalUpdated, err := metalOptimizer.Step(metalState)
				So(err, ShouldBeNil)

				assertMetalSlice(
					fmt.Sprintf("lbfgs/out/%d", length),
					metalUpdated.Out,
					cpuUpdated.Out,
					1e-4,
				)
				So(metalState.Head, ShouldEqual, cpuState.Head)
				So(metalState.Count, ShouldEqual, cpuState.Count)
			}
		})
	})
}

func BenchmarkLBFGS_Step(benchmark *testing.B) {
	optimizer, err := NewOptimizerRegistry().LBFGS(state.NewDict())
	if err != nil {
		benchmark.Fatal(err)
	}

	stateDict := lbfgsState(
		pattern(8192, 0.41, 0.017),
		pattern(8192, -0.23, 0.011),
	)
	nextGrads := pattern(8192, -0.18, 0.008)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		_, _ = optimizer.Step(stateDict)
		stateDict.WithParams(stateDict.Out).WithGrads(nextGrads)
	}
}

func BenchmarkOptimizerReduction_Step(benchmark *testing.B) {
	for _, optimizerCase := range metalOptimizerCases() {
		if optimizerCase.name != "lars" && optimizerCase.name != "lamb" {
			continue
		}

		benchmark.Run(optimizerCase.name, func(benchmark *testing.B) {
			optimizer, err := optimizerCase.metal(NewOptimizerRegistry())
			if err != nil {
				benchmark.Fatal(err)
			}

			stateDict := optimizerState(1 << 16)

			benchmark.ResetTimer()

			for benchmark.Loop() {
				updated, err := optimizer.Step(stateDict)
				if err != nil {
					benchmark.Fatal(err)
				}

				stateDict.WithParams(updated.Out)
			}
		})
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

type metalOptimizerCase struct {
	name      string
	fields    []string
	tolerance float64
	cpu       func() state.Optimizer
	metal     func(Registry) (state.Optimizer, error)
}

func metalOptimizerCases() []metalOptimizerCase {
	tolerance := 8e-4

	return []metalOptimizerCase{
		{
			name:      "adam",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdam() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.Adam(state.NewDict()) },
		},
		{
			name:      "adamw",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdamW() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.AdamW(state.NewDict()) },
		},
		{
			name:      "adamax",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdaMax() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.AdaMax(state.NewDict()) },
		},
		{
			name:      "sgd",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpusgd.NewSGD() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.SGD(state.NewDict()) },
		},
		{
			name:      "lion",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulion.NewLion() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.Lion(state.NewDict()) },
		},
		{
			name:      "rmsprop",
			fields:    []string{"v", "buf", "grad_avg"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpurmsprop.NewRMSProp() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.RMSProp(state.NewDict()) },
		},
		{
			name:      "hebbian",
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuhebbian.NewHebbian() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.Hebbian(state.NewDict()) },
		},
		{
			name:      "lars",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulars.NewLARS() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.Lars(state.NewDict()) },
		},
		{
			name:      "lamb",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulars.NewLAMB() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.Lamb(state.NewDict()) },
		},
		{
			name:      "adagrad",
			fields:    []string{"v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadagrad.NewAdaGrad() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.AdaGrad(state.NewDict()) },
		},
		{
			name:      "adadelta",
			fields:    []string{"v", "buf"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadagrad.NewAdaDelta() },
			metal:     func(registry Registry) (state.Optimizer, error) { return registry.AdaDelta(state.NewDict()) },
		},
	}
}

func optimizerState(length int) *state.Dict {
	stateDict := state.NewDict().
		WithLR(0.0125).
		WithBeta1(0.86).
		WithBeta2(0.997).
		WithEps(1e-6).
		WithWD(0.015).
		WithMomentum(0.72).
		WithAlpha(0.93).
		WithRho(0.91).
		WithEta(0.001).
		WithLRDecay(0.02).
		WithMaxNorm(17.0).
		WithNesterov(true).
		WithCentered(true).
		WithParams(pattern(length, 0.17, 0.031)).
		WithGrads(pattern(length, -0.11, 0.023)).
		WithM(pattern(length, 0.013, 0.007)).
		WithV(positivePattern(length, 0.021, 0.005))

	stateDict.Buf = positivePattern(length, 0.019, 0.004)
	stateDict.GradAvg = pattern(length, -0.015, 0.003)

	return stateDict
}

func pattern(length int, offset float64, step float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		sign := 1.0

		if index%2 != 0 {
			sign = -1.0
		}

		values[index] = sign*(offset+step*float64(index%11)) + 0.0007*float64(index/11)
	}

	return values
}

func positivePattern(length int, offset float64, step float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = offset + step*float64(index%11) + 0.0007*float64(index/11)
	}

	return values
}

func lbfgsShift(values []float64, direction []float64, scale float64) []float64 {
	shifted := make([]float64, len(values))

	for index := range values {
		shifted[index] = values[index] + scale*direction[index]
	}

	return shifted
}

func stateField(stateDict *state.Dict, field string) []float64 {
	switch field {
	case "m":
		return stateDict.M
	case "v":
		return stateDict.V
	case "buf":
		return stateDict.Buf
	case "grad_avg":
		return stateDict.GradAvg
	default:
		return nil
	}
}

func assertMetalSlice(name string, actual []float64, expected []float64, tolerance float64) {
	So(actual, ShouldHaveLength, len(expected))

	if len(expected) == 0 {
		return
	}

	maxDiff := 0.0
	maxIndex := 0

	for index := range expected {
		if math.IsNaN(expected[index]) || math.IsNaN(actual[index]) {
			SoMsg(name, math.IsNaN(expected[index]), ShouldBeTrue)
			SoMsg(name, math.IsNaN(actual[index]), ShouldBeTrue)
			continue
		}

		diff := math.Abs(actual[index] - expected[index])

		if diff <= maxDiff {
			continue
		}

		maxDiff = diff
		maxIndex = index
	}

	SoMsg(
		fmt.Sprintf("%s max_diff=%g index=%d actual=%g expected=%g tolerance=%g",
			name, maxDiff, maxIndex, actual[maxIndex], expected[maxIndex], tolerance,
		),
		maxDiff <= tolerance,
		ShouldBeTrue,
	)
}
