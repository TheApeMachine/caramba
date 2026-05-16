package convolution

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestConv1d_Forward(t *testing.T) {
	Convey("Given a Conv1d operation", t, func() {
		operation := NewConv1d(4, 4, 3, 1, 1, 1, 2)

		Convey("Forward", func() {
			Convey("It should match the reference grouped convolution", func() {
				batch := 2
				inChannels := 4
				length := 9
				outChannels := 4
				kernelSize := 3
				stride := 1
				padding := 1
				dilation := 1
				groups := 2
				input := convolutionSequence(batch*inChannels*length, 0.05, -0.2)
				weight := convolutionSequence(outChannels*(inChannels/groups)*kernelSize, 0.03, 0.1)
				bias := convolutionSequence(outChannels, 0.02, -0.01)
				stateDict := state.NewDict().
					WithShape([]int{batch, inChannels, length}).
					WithInput(input).
					WithWeight(weight).
					WithBias(bias)
				stateDict.InChannels = inChannels
				stateDict.OutChannels = outChannels
				stateDict.KernelSize = kernelSize
				stateDict.Stride = stride
				stateDict.Padding = padding
				stateDict.Dilation = dilation
				stateDict.Groups = groups

				outputState, err := operation.Forward(stateDict)
				expected := referenceConv1d(
					input, weight, bias,
					batch, inChannels, length,
					outChannels, kernelSize, stride, padding, dilation, groups,
				)

				So(err, ShouldBeNil)
				assertConvolutionAlmostEqual(outputState.Out, expected)
			})
		})
	})
}

func BenchmarkConv1d_Forward(benchmark *testing.B) {
	operation := NewConv1d(16, 32, 5, 1, 2, 1, 1)
	input := convolutionSequence(4*16*128, 0.01, 0.0)
	weight := convolutionSequence(32*16*5, 0.005, 0.0)
	bias := convolutionSequence(32, 0.001, 0.0)

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{4, 16, 128}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias)
		stateDict.InChannels = 16
		stateDict.OutChannels = 32
		stateDict.KernelSize = 5
		stateDict.Stride = 1
		stateDict.Padding = 2
		stateDict.Dilation = 1
		stateDict.Groups = 1
		_, _ = operation.Forward(stateDict)
	}
}
