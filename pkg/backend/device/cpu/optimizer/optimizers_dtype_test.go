package optimizer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Mixed-precision optimizer dispatch parity. Validates that the bf16/fp16
mixed-precision Adam step (params bf16/fp16, gradients bf16/fp16, state
f32, output bf16/fp16) reaches an output within standard ULP bounds of
the f32 reference applied to bf16-rounded inputs.
*/

func TestAdamStepBFloat16Dispatch(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(0xADAB))

	// Set up bf16 params + gradients, f32 moments.
	paramsT, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.BFloat16)
	gradT, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.BFloat16)
	firstT, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float32)
	secondT, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float32)
	outT, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.BFloat16)

	paramsView, _ := paramsT.BFloat16Native()
	gradView, _ := gradT.BFloat16Native()
	firstView, _ := firstT.Float32Native()
	secondView, _ := secondT.Float32Native()

	for index := range n {
		paramsView[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()) * 0.1)
		gradView[index] = dtype.NewBfloat16FromFloat32(float32(rng.NormFloat64()) * 0.01)
		firstView[index] = float32(rng.NormFloat64()) * 0.01
		secondView[index] = float32(rng.NormFloat64()*rng.NormFloat64()) * 0.001
	}

	// Reference: widen inputs, run f32 adam_step, compare narrowed result.
	refParams := make([]float32, n)
	refGrad := make([]float32, n)
	refFirst := make([]float32, n)
	refSecond := make([]float32, n)
	refOut := make([]float32, n)

	for index := range n {
		refParams[index] = (&paramsView[index]).Float32()
		refGrad[index] = (&gradView[index]).Float32()
		refFirst[index] = firstView[index]
		refSecond[index] = secondView[index]
	}

	adamStepSlices(DefaultAdamConfig(), refParams, refGrad, refFirst, refSecond, refOut)

	// Run the bf16 mixed-precision dispatch.
	if err := runMixedOptimizerBFloat16(
		[]tensor.Tensor{paramsT, gradT, firstT, secondT, outT},
		2,
		func(params, gradients []float32, state [][]float32, output []float32) {
			adamStepSlices(DefaultAdamConfig(), params, gradients, state[0], state[1], output)
		},
	); err != nil {
		t.Fatal(err)
	}

	outView, _ := outT.BFloat16Native()

	for index := range n {
		expected := dtype.NewBfloat16FromFloat32(refOut[index])
		expectedBits := uint16(expected)
		actualBits := uint16(outView[index])

		if expectedBits == actualBits {
			continue
		}

		diff := int(expectedBits) - int(actualBits)
		if diff < 0 {
			diff = -diff
		}

		// 2 ULPs of bf16 — accommodates the ordering difference where
		// the reference reads moments BEFORE the mixed-precision pass
		// writes them. Both paths share the same slice math, so the
		// state-update divergence is null and the only ULP source is
		// f32→bf16 narrowing of the param update.
		if diff > 2 {
			t.Fatalf("lane %d expected=0x%04x (%g) actual=0x%04x (%g) ulp=%d",
				index, expectedBits, expected.Float32(),
				actualBits, outView[index].Float32(), diff,
			)
		}
	}
}

// suppress unused-import warning for math
var _ = math.Sqrt
