//go:build darwin && cgo

package metal

import (
	"testing"

	cpuattention "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	cpupositional "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/positional"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMetalPositional_RoPEForward(test *testing.T) {
	lib := metallibPathOrSkip(test, "positional.metallib")

	Convey("Given Metal RoPE", test, func() {
		positionalOps, err := NewPositional(lib)
		So(err, ShouldBeNil)

		Convey("It should match CPU RoPE for head-major tensors", func() {
			input := positionalSequence(1 * 2 * 3 * 4)
			shape := []int{1, 2, 3, 4}
			expectedState, err := cpupositional.NewRoPE().Forward(
				state.NewDict().WithShape(shape).WithInput(input),
			)
			So(err, ShouldBeNil)

			actual, err := positionalOps.RoPEForward(10000, shape, input)

			So(err, ShouldBeNil)
			assertAlmostEqualSlice(actual, expectedState.Out, 1e-4)
		})

		Convey("It should reject projection tensors before head shaping", func() {
			_, err := positionalOps.RoPEForward(
				10000,
				[]int{1, 3, 8},
				make([]float64, 24),
			)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "expected rank 4")
		})
	})
}

func TestMetalAttention_GQA(test *testing.T) {
	lib := metallibPathOrSkip(test, "attention.metallib")

	Convey("Given Metal GQA", test, func() {
		attentionOps, err := NewAttention(lib)
		So(err, ShouldBeNil)

		Convey("It should match CPU causal GQA for rank-four head tensors", func() {
			query := positionalSequence(1 * 2 * 3 * 4)
			key := positionalSequence(1 * 1 * 3 * 4)
			value := positionalSequence(1 * 1 * 3 * 4)
			expectedDict := state.NewDict().WithShape([]int{1, 2, 3, 4}).WithInputs(
				query,
				key,
				value,
			)
			expectedDict.NumKVHeads = 1
			expectedDict.HeadDim = 4
			expectedDict.Causal = true
			expectedState, err := cpuattention.NewGQA().Forward(expectedDict)
			So(err, ShouldBeNil)

			actual, err := attentionOps.GQA(query, key, value, 1, 2, 1, 3, 4, true)

			So(err, ShouldBeNil)
			assertAlmostEqualSlice(actual, expectedState.Out, 1e-4)
		})
	})
}

func positionalSequence(length int) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = float64(index%17) / 17
	}

	return values
}
