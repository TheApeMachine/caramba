package convolution

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestConv3d_Forward(t *testing.T) {
	Convey("Given a Conv3d operation", t, func() {
		operation := NewConv3d(2, 2, 2, 2, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1)

		Convey("Forward", func() {
			Convey("It should match the reference convolution", func() {
				batch := 1
				inChannels := 2
				depth := 4
				height := 4
				width := 5
				outChannels := 2
				kernelD := 2
				kernelH := 2
				kernelW := 3
				strideD := 1
				strideH := 1
				strideW := 1
				padD := 0
				padH := 0
				padW := 1
				dilationD := 1
				dilationH := 1
				dilationW := 1
				groups := 1
				input := convolutionSequence(batch*inChannels*depth*height*width, 0.025, -0.04)
				weight := convolutionSequence(outChannels*inChannels*kernelD*kernelH*kernelW, 0.018, 0.02)
				bias := convolutionSequence(outChannels, 0.013, -0.02)
				stateDict := state.NewDict().
					WithShape([]int{batch, inChannels, depth, height, width}).
					WithInput(input).
					WithWeight(weight).
					WithBias(bias)
				stateDict.InChannels = inChannels
				stateDict.OutChannels = outChannels
				stateDict.KernelD = kernelD
				stateDict.KernelH = kernelH
				stateDict.KernelW = kernelW
				stateDict.StrideD = strideD
				stateDict.StrideH = strideH
				stateDict.StrideW = strideW
				stateDict.PadD = padD
				stateDict.PadH = padH
				stateDict.PadW = padW
				stateDict.DilationD = dilationD
				stateDict.DilationH = dilationH
				stateDict.DilationW = dilationW
				stateDict.Groups = groups

				outputState, err := operation.Forward(stateDict)
				expected := referenceConv3d(
					input, weight, bias,
					batch, inChannels, depth, height, width,
					outChannels, kernelD, kernelH, kernelW,
					strideD, strideH, strideW,
					padD, padH, padW,
					dilationD, dilationH, dilationW, groups,
				)

				So(err, ShouldBeNil)
				assertConvolutionAlmostEqual(outputState.Out, expected)
			})
		})
	})
}

func BenchmarkConv3d_Forward(benchmark *testing.B) {
	operation := NewConv3d(8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
	input := convolutionSequence(1*8*12*12*12, 0.01, 0.0)
	weight := convolutionSequence(8*8*3*3*3, 0.005, 0.0)
	bias := convolutionSequence(8, 0.001, 0.0)

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1, 8, 12, 12, 12}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias)
		stateDict.InChannels = 8
		stateDict.OutChannels = 8
		stateDict.KernelD = 3
		stateDict.KernelH = 3
		stateDict.KernelW = 3
		stateDict.StrideD = 1
		stateDict.StrideH = 1
		stateDict.StrideW = 1
		stateDict.PadD = 1
		stateDict.PadH = 1
		stateDict.PadW = 1
		stateDict.DilationD = 1
		stateDict.DilationH = 1
		stateDict.DilationW = 1
		stateDict.Groups = 1
		_, _ = operation.Forward(stateDict)
	}
}
