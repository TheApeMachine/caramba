package convolution

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestConv2d_Forward(t *testing.T) {
	Convey("Given a Conv2d operation", t, func() {
		operation := NewConv2d(4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 2)

		Convey("Forward", func() {
			Convey("It should match the reference grouped convolution", func() {
				batch := 1
				inChannels := 4
				height := 5
				width := 6
				outChannels := 4
				kernelH := 3
				kernelW := 3
				strideH := 1
				strideW := 1
				padH := 1
				padW := 1
				dilationH := 1
				dilationW := 1
				groups := 2
				input := convolutionSequence(batch*inChannels*height*width, 0.04, 0.03)
				weight := convolutionSequence(outChannels*(inChannels/groups)*kernelH*kernelW, 0.02, -0.07)
				bias := convolutionSequence(outChannels, 0.015, 0.02)
				stateDict := state.NewDict().
					WithShape([]int{batch, inChannels, height, width}).
					WithInput(input).
					WithWeight(weight).
					WithBias(bias)
				stateDict.InChannels = inChannels
				stateDict.OutChannels = outChannels
				stateDict.KernelH = kernelH
				stateDict.KernelW = kernelW
				stateDict.StrideH = strideH
				stateDict.StrideW = strideW
				stateDict.PadH = padH
				stateDict.PadW = padW
				stateDict.DilationH = dilationH
				stateDict.DilationW = dilationW
				stateDict.Groups = groups

				outputState, err := operation.Forward(stateDict)
				expected := referenceConv2d(
					input, weight, bias,
					batch, inChannels, height, width,
					outChannels, kernelH, kernelW,
					strideH, strideW, padH, padW,
					dilationH, dilationW, groups,
				)

				So(err, ShouldBeNil)
				assertConvolutionAlmostEqual(outputState.Out, expected)
			})
		})
	})
}

func BenchmarkConv2d_Forward(benchmark *testing.B) {
	operation := NewConv2d(16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1)
	input := convolutionSequence(2*16*32*32, 0.01, 0.0)
	weight := convolutionSequence(32*16*3*3, 0.005, 0.0)
	bias := convolutionSequence(32, 0.001, 0.0)

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{2, 16, 32, 32}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias)
		stateDict.InChannels = 16
		stateDict.OutChannels = 32
		stateDict.KernelH = 3
		stateDict.KernelW = 3
		stateDict.StrideH = 1
		stateDict.StrideW = 1
		stateDict.PadH = 1
		stateDict.PadW = 1
		stateDict.DilationH = 1
		stateDict.DilationW = 1
		stateDict.Groups = 1
		_, _ = operation.Forward(stateDict)
	}
}
