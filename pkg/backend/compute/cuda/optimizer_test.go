//go:build linux && cgo && cuda

package cuda

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
	Convey("Given CUDA optimizers for the full train optimizer contract", test, func() {
		for _, optimizerCase := range cudaOptimizerCases() {
			optimizerCase := optimizerCase

			Convey("It should match CPU state-dict execution for "+optimizerCase.name, func() {
				for _, length := range []int{1, 7, 64, 1024, 8192} {
					cpuState := cudaOptimizerState(length)
					cudaState := cudaOptimizerState(length)
					cpuOptimizer := optimizerCase.cpu()
					cudaOptimizer, err := optimizerCase.cuda(NewOptimizerRegistry())

					So(err, ShouldBeNil)

					cpuUpdated, err := cpuOptimizer.Step(cpuState)
					So(err, ShouldBeNil)
					cudaUpdated, err := cudaOptimizer.Step(cudaState)
					So(err, ShouldBeNil)

					assertCUDASlice(
						fmt.Sprintf("%s/out/%d", optimizerCase.name, length),
						cudaUpdated.Out,
						cpuUpdated.Out,
						optimizerCase.tolerance,
					)

					for _, field := range optimizerCase.fields {
						assertCUDASlice(
							fmt.Sprintf("%s/%s/%d", optimizerCase.name, field, length),
							cudaStateField(cudaUpdated, field),
							cudaStateField(cpuUpdated, field),
							optimizerCase.tolerance,
						)
					}

					So(cudaUpdated.Step, ShouldEqual, cpuUpdated.Step)
				}
			})
		}
	})
}

func TestLBFGS_Step(test *testing.T) {
	Convey("Given a CUDA L-BFGS optimizer", test, func() {
		cudaOptimizer, err := NewOptimizerRegistry().LBFGS(state.NewDict())
		cpuOptimizer := cpulbfgs.NewLBFGS()

		So(err, ShouldBeNil)

		Convey("It should match the CPU two-loop update across backend kernel lengths", func() {
			for _, length := range []int{1, 7, 64, 1024, 8192} {
				cpuState := cudaLBFGSState(
					cudaPattern(length, 0.19, 0.027),
					cudaPattern(length, -0.07, 0.019),
				)
				cudaState := cudaLBFGSState(
					cudaPattern(length, 0.19, 0.027),
					cudaPattern(length, -0.07, 0.019),
				)

				_, err = cpuOptimizer.Step(cpuState)
				So(err, ShouldBeNil)
				_, err = cudaOptimizer.Step(cudaState)
				So(err, ShouldBeNil)

				cpuState.WithParams(cudaPattern(length, 0.13, 0.021)).
					WithGrads(cudaPattern(length, -0.05, 0.017))
				cudaState.WithParams(cudaPattern(length, 0.13, 0.021)).
					WithGrads(cudaPattern(length, -0.05, 0.017))

				cpuUpdated, err := cpuOptimizer.Step(cpuState)
				So(err, ShouldBeNil)
				cudaUpdated, err := cudaOptimizer.Step(cudaState)
				So(err, ShouldBeNil)

				assertCUDASlice(fmt.Sprintf("lbfgs/out/%d", length), cudaUpdated.Out, cpuUpdated.Out, 1e-9)
				So(cudaState.Head, ShouldEqual, cpuState.Head)
				So(cudaState.Count, ShouldEqual, cpuState.Count)
			}
		})
	})
}

func BenchmarkLBFGS_Step(benchmark *testing.B) {
	optimizer, err := NewOptimizerRegistry().LBFGS(state.NewDict())
	if err != nil {
		benchmark.Fatal(err)
	}

	stateDict := cudaLBFGSState(
		cudaPattern(1<<16, 0.19, 0.027),
		cudaPattern(1<<16, -0.07, 0.019),
	)
	_, err = optimizer.Step(stateDict)
	if err != nil {
		benchmark.Fatal(err)
	}

	stateDict.WithParams(cudaPattern(1<<16, 0.13, 0.021)).
		WithGrads(cudaPattern(1<<16, -0.05, 0.017))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		updated, err := optimizer.Step(stateDict)
		if err != nil {
			benchmark.Fatal(err)
		}

		stateDict.WithParams(updated.Out)
	}
}

func BenchmarkOptimizerReduction_Step(benchmark *testing.B) {
	for _, optimizerCase := range cudaOptimizerCases() {
		if optimizerCase.name != "lars" && optimizerCase.name != "lamb" {
			continue
		}

		benchmark.Run(optimizerCase.name, func(benchmark *testing.B) {
			optimizer, err := optimizerCase.cuda(NewOptimizerRegistry())
			if err != nil {
				benchmark.Fatal(err)
			}

			stateDict := cudaOptimizerState(1 << 16)

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

type cudaOptimizerCase struct {
	name      string
	fields    []string
	tolerance float64
	cpu       func() state.Optimizer
	cuda      func(Registry) (state.Optimizer, error)
}

func cudaOptimizerCases() []cudaOptimizerCase {
	tolerance := 1e-8

	return []cudaOptimizerCase{
		{
			name:      "adam",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdam() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.Adam(state.NewDict()) },
		},
		{
			name:      "adamw",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdamW() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.AdamW(state.NewDict()) },
		},
		{
			name:      "adamax",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadam.NewAdaMax() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.AdaMax(state.NewDict()) },
		},
		{
			name:      "sgd",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpusgd.NewSGD() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.SGD(state.NewDict()) },
		},
		{
			name:      "lion",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulion.NewLion() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.Lion(state.NewDict()) },
		},
		{
			name:      "rmsprop",
			fields:    []string{"v", "buf", "grad_avg"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpurmsprop.NewRMSProp() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.RMSProp(state.NewDict()) },
		},
		{
			name:      "hebbian",
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuhebbian.NewHebbian() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.Hebbian(state.NewDict()) },
		},
		{
			name:      "lars",
			fields:    []string{"m"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulars.NewLARS() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.Lars(state.NewDict()) },
		},
		{
			name:      "lamb",
			fields:    []string{"m", "v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpulars.NewLAMB() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.Lamb(state.NewDict()) },
		},
		{
			name:      "adagrad",
			fields:    []string{"v"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadagrad.NewAdaGrad() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.AdaGrad(state.NewDict()) },
		},
		{
			name:      "adadelta",
			fields:    []string{"v", "buf"},
			tolerance: tolerance,
			cpu:       func() state.Optimizer { return cpuadagrad.NewAdaDelta() },
			cuda:      func(registry Registry) (state.Optimizer, error) { return registry.AdaDelta(state.NewDict()) },
		},
	}
}

func cudaOptimizerState(length int) *state.Dict {
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
		WithParams(cudaPattern(length, 0.17, 0.031)).
		WithGrads(cudaPattern(length, -0.11, 0.023)).
		WithM(cudaPattern(length, 0.013, 0.007)).
		WithV(cudaPositivePattern(length, 0.021, 0.005))

	stateDict.Buf = cudaPositivePattern(length, 0.019, 0.004)
	stateDict.GradAvg = cudaPattern(length, -0.015, 0.003)

	return stateDict
}

func cudaLBFGSState(params, grads []float64) *state.Dict {
	return state.NewDict().
		WithLR(0.4).
		WithHistSize(4).
		WithLineSearch(false).
		WithC1(1e-4).
		WithParams(params).
		WithGrads(grads)
}

func cudaPattern(length int, offset float64, step float64) []float64 {
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

func cudaPositivePattern(length int, offset float64, step float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = offset + step*float64(index%11) + 0.0007*float64(index/11)
	}

	return values
}

func cudaStateField(stateDict *state.Dict, field string) []float64 {
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

func assertCUDASlice(name string, actual []float64, expected []float64, tolerance float64) {
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
