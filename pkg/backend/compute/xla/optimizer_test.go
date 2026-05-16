//go:build cgo && xla

package xla

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
	platform := xlaPJRTAvailablePlatform(test)

	Convey("Given XLA optimizers for the full train optimizer contract", test, func() {
		registry := NewOptimizerRegistryForPlatform(platform)

		for _, optimizerCase := range xlaOptimizerCases() {
			optimizerCase := optimizerCase

			Convey("It should match CPU state-dict execution for "+optimizerCase.name, func() {
				for _, length := range []int{1, 7, 64, 1024, 8192} {
					cpuState := xlaOptimizerState(length)
					xlaState := xlaOptimizerState(length)
					cpuOptimizer := optimizerCase.cpu()
					xlaOptimizer, err := optimizerCase.xla(registry)

					So(err, ShouldBeNil)

					cpuUpdated, err := cpuOptimizer.Step(cpuState)
					So(err, ShouldBeNil)
					xlaUpdated, err := xlaOptimizer.Step(xlaState)
					So(err, ShouldBeNil)

					assertXLASlice(
						fmt.Sprintf("%s/out/%d", optimizerCase.name, length),
						xlaUpdated.Out,
						cpuUpdated.Out,
						optimizerCase.tolerance,
					)

					for _, field := range optimizerCase.fields {
						assertXLASlice(
							fmt.Sprintf("%s/%s/%d", optimizerCase.name, field, length),
							xlaStateField(xlaUpdated, field),
							xlaStateField(cpuUpdated, field),
							optimizerCase.tolerance,
						)
					}

					So(xlaUpdated.Step, ShouldEqual, cpuUpdated.Step)
				}
			})
		}
	})
}

func TestLBFGS_Step(test *testing.T) {
	platform := xlaPJRTAvailablePlatform(test)

	Convey("Given a XLA L-BFGS optimizer", test, func() {
		xlaOptimizer, err := NewOptimizerRegistryForPlatform(platform).LBFGS(state.NewDict())
		cpuOptimizer := cpulbfgs.NewLBFGS()

		So(err, ShouldBeNil)

		Convey("It should match the CPU two-loop update after history is populated", func() {
			cpuState := xlaLBFGSState(
				[]float64{1.0, -0.5, 0.25, 1.5},
				[]float64{0.30, -0.10, 0.20, -0.40},
			)
			xlaState := xlaLBFGSState(
				[]float64{1.0, -0.5, 0.25, 1.5},
				[]float64{0.30, -0.10, 0.20, -0.40},
			)

			_, err = cpuOptimizer.Step(cpuState)
			So(err, ShouldBeNil)
			_, err = xlaOptimizer.Step(xlaState)
			So(err, ShouldBeNil)

			cpuState.WithParams([]float64{0.92, -0.42, 0.18, 1.36}).
				WithGrads([]float64{0.18, -0.04, 0.12, -0.22})
			xlaState.WithParams([]float64{0.92, -0.42, 0.18, 1.36}).
				WithGrads([]float64{0.18, -0.04, 0.12, -0.22})

			cpuUpdated, err := cpuOptimizer.Step(cpuState)
			So(err, ShouldBeNil)
			xlaUpdated, err := xlaOptimizer.Step(xlaState)
			So(err, ShouldBeNil)

			assertXLASlice("lbfgs/out", xlaUpdated.Out, cpuUpdated.Out, 1e-9)
			So(xlaState.Head, ShouldEqual, cpuState.Head)
			So(xlaState.Count, ShouldEqual, cpuState.Count)
		})
	})
}

func BenchmarkOptimizerReduction_Step(benchmark *testing.B) {
	platform := xlaPJRTAvailablePlatform(benchmark)
	registry := NewOptimizerRegistryForPlatform(platform)

	for _, optimizerCase := range xlaOptimizerCases() {
		if optimizerCase.name != "lars" && optimizerCase.name != "lamb" {
			continue
		}

		benchmark.Run(optimizerCase.name, func(benchmark *testing.B) {
			optimizer, err := optimizerCase.xla(registry)
			if err != nil {
				benchmark.Fatal(err)
			}

			stateDict := xlaOptimizerState(1 << 16)

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

type xlaOptimizerCase struct {
	name      string
	fields    []string
	tolerance float64
	cpu       func() state.Optimizer
	xla       func(Registry) (state.Optimizer, error)
}

func xlaOptimizerCases() []xlaOptimizerCase {
	tolerance := 1e-8

	return []xlaOptimizerCase{
		{
			name:      "adam",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdam() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.Adam(state.NewDict()) },
		},
		{
			name:      "adamw",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdamW() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.AdamW(state.NewDict()) },
		},
		{
			name:      "adamax",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdaMax() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.AdaMax(state.NewDict()) },
		},
		{
			name:      "sgd",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpusgd.NewSGD() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.SGD(state.NewDict()) },
		},
		{
			name:      "lion",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulion.NewLion() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.Lion(state.NewDict()) },
		},
		{
			name:      "rmsprop",
			fields:    []string{"v", "buf", "grad_avg"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpurmsprop.NewRMSProp() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.RMSProp(state.NewDict()) },
		},
		{
			name:      "hebbian",
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuhebbian.NewHebbian() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.Hebbian(state.NewDict()) },
		},
		{
			name:      "lars",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulars.NewLARS() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.Lars(state.NewDict()) },
		},
		{
			name:      "lamb",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulars.NewLAMB() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.Lamb(state.NewDict()) },
		},
		{
			name:      "adagrad",
			fields:    []string{"v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadagrad.NewAdaGrad() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.AdaGrad(state.NewDict()) },
		},
		{
			name:      "adadelta",
			fields:    []string{"v", "buf"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadagrad.NewAdaDelta() },
			xla:       func(registry Registry) (state.Optimizer, error) { return registry.AdaDelta(state.NewDict()) },
		},
	}
}

func xlaOptimizerState(length int) *state.Dict {
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
		WithParams(xlaPattern(length, 0.17, 0.031)).
		WithGrads(xlaPattern(length, -0.11, 0.023)).
		WithM(xlaPattern(length, 0.013, 0.007)).
		WithV(xlaPositivePattern(length, 0.021, 0.005))

	stateDict.Buf = xlaPositivePattern(length, 0.019, 0.004)
	stateDict.GradAvg = xlaPattern(length, -0.015, 0.003)

	return stateDict
}

func xlaLBFGSState(params, grads []float64) *state.Dict {
	return state.NewDict().
		WithLR(0.4).
		WithHistSize(4).
		WithLineSearch(false).
		WithC1(1e-4).
		WithParams(params).
		WithGrads(grads)
}

func xlaPattern(length int, offset float64, step float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		sign := 1.0

		if index%2 == 1 {
			sign = -1.0
		}

		values[index] = sign * (offset + float64(index%17)*step)
	}

	return values
}

func xlaPositivePattern(length int, offset float64, step float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = offset + float64(index%19)*step
	}

	return values
}

func xlaStateField(stateDict *state.Dict, field string) []float64 {
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

func assertXLASlice(name string, actual []float64, expected []float64, tolerance float64) {
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
		fmt.Sprintf(
			"%s max_diff=%g index=%d actual=%g expected=%g tolerance=%g",
			name,
			maxDiff,
			maxIndex,
			actual[maxIndex],
			expected[maxIndex],
			tolerance,
		),
		maxDiff <= tolerance,
		ShouldBeTrue,
	)
}
