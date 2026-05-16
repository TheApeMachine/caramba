package convolution

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestConvTranspose2d_Forward(t *testing.T) {
	Convey("Given a ConvTranspose2d operation", t, func() {
		operation := NewConvTranspose2d(2, 4, 3, 3, 2, 2, 0, 0, 0, 0, 1, 1, 2)

		Convey("Forward", func() {
			Convey("It should match the reference grouped transposed convolution", func() {
				batch := 1
				inChannels := 2
				height := 3
				width := 4
				outChannels := 4
				kernelH := 3
				kernelW := 3
				strideH := 2
				strideW := 2
				groups := 2
				input := convolutionSequence(batch*inChannels*height*width, 0.04, -0.03)
				weight := convolutionSequence(inChannels*(outChannels/groups)*kernelH*kernelW, 0.021, 0.01)
				bias := convolutionSequence(outChannels, 0.014, -0.01)
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
				stateDict.PadH = 0
				stateDict.PadW = 0
				stateDict.OutPadH = 0
				stateDict.OutPadW = 0
				stateDict.DilationH = 1
				stateDict.DilationW = 1
				stateDict.Groups = groups

				outputState, err := operation.Forward(stateDict)
				expected := referenceConvTranspose2d(
					input, weight, bias,
					batch, inChannels, height, width,
					outChannels, kernelH, kernelW,
					strideH, strideW, groups,
				)

				So(err, ShouldBeNil)
				assertConvolutionAlmostEqual(outputState.Out, expected)
			})
		})
	})
}

func BenchmarkConvTranspose2d_Forward(benchmark *testing.B) {
	operation := NewConvTranspose2d(8, 8, 3, 3, 2, 2, 0, 0, 0, 0, 1, 1, 1)
	input := convolutionSequence(1*8*16*16, 0.01, 0.0)
	weight := convolutionSequence(8*8*3*3, 0.005, 0.0)
	bias := convolutionSequence(8, 0.001, 0.0)

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1, 8, 16, 16}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias)
		stateDict.InChannels = 8
		stateDict.OutChannels = 8
		stateDict.KernelH = 3
		stateDict.KernelW = 3
		stateDict.StrideH = 2
		stateDict.StrideW = 2
		stateDict.PadH = 0
		stateDict.PadW = 0
		stateDict.OutPadH = 0
		stateDict.OutPadW = 0
		stateDict.DilationH = 1
		stateDict.DilationW = 1
		stateDict.Groups = 1
		_, _ = operation.Forward(stateDict)
	}
}
