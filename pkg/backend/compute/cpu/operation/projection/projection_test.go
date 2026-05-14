package projection

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLinear(t *testing.T) {
	Convey("Given a Linear operation", t, func() {
		op := NewLinear()

		Convey("Forward", func() {
			Convey("It should compute x @ weight + bias from the state dict", func() {
				stateDict := state.NewDict().
					WithShape([]int{2, 3}).
					WithInput([]float64{1, 2, 3, 4, 5, 6}).
					WithWeight([]float64{1, 10, 2, 20, 3, 30}).
					WithBias([]float64{1, -1})
				stateDict.InFeatures = 3
				stateDict.OutFeatures = 2

				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{15, 139, 33, 319})
			})
		})
	})
}

func BenchmarkLinear_Forward(b *testing.B) {
	op := NewLinear()
	M, K, N := 64, 128, 256
	input := make([]float64, M*K)
	weight := make([]float64, K*N)
	bias := make([]float64, N)

	for index := range input {
		input[index] = float64(index%17) * 0.01
	}

	for index := range weight {
		weight[index] = float64(index%13) * 0.02
	}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{M, K}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias)
		stateDict.InFeatures = K
		stateDict.OutFeatures = N
		_, _ = op.Forward(stateDict)
	}
}

func TestFusedQKV(t *testing.T) {
	Convey("Given a FusedQKV operation", t, func() {
		op := NewFusedQKV()

		Convey("Forward", func() {
			Convey("It should compute the fused projection from the state dict", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 2}).
					WithInput([]float64{1, 2}).
					WithWeight([]float64{1, 10, 100, 2, 20, 200}).
					WithBias([]float64{1, 2, 3})
				stateDict.DIn = 2
				stateDict.DQ = 1
				stateDict.DK = 1
				stateDict.DV = 1

				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{6, 52, 503})
			})
		})
	})
}

func BenchmarkFusedQKV_Forward(b *testing.B) {
	op := NewFusedQKV()
	M, K, N := 64, 128, 384
	input := make([]float64, M*K)
	weight := make([]float64, K*N)
	bias := make([]float64, N)

	for index := range input {
		input[index] = float64(index%19) * 0.01
	}

	for index := range weight {
		weight[index] = float64(index%23) * 0.02
	}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{M, K}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias)
		stateDict.DIn = K
		stateDict.DQ = 128
		stateDict.DK = 128
		stateDict.DV = 128
		_, _ = op.Forward(stateDict)
	}
}

func TestTiedEmbedding(t *testing.T) {
	Convey("Given a TiedEmbedding operation", t, func() {
		op, err := NewTiedEmbedding(nil, 0, 0)
		So(err, ShouldBeNil)

		Convey("Forward", func() {
			Convey("It should compute logits from the state dict", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 2, 2}).
					WithInput([]float64{1, 2, 3, 4}).
					WithWeight([]float64{1, 10, 100, 2, 20, 200})
				stateDict.DModel = 2
				stateDict.VocabSize = 3

				outputState, err := op.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{5, 50, 500, 11, 110, 1100})
			})
		})
	})
}

func BenchmarkTiedEmbedding_Forward(b *testing.B) {
	op, _ := NewTiedEmbedding(nil, 0, 0)
	M, K, N := 128, 256, 4096
	input := make([]float64, M*K)
	weight := make([]float64, K*N)

	for index := range input {
		input[index] = float64(index%29) * 0.01
	}

	for index := range weight {
		weight[index] = float64(index%31) * 0.001
	}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1, M, K}).
			WithInput(input).
			WithWeight(weight)
		stateDict.DModel = K
		stateDict.VocabSize = N
		_, _ = op.Forward(stateDict)
	}
}
