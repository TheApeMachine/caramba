//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestConvolutionOps_ConvTranspose2d(test *testing.T) {
	lib := metallibPathOrSkip(test, "convolution.metallib")

	Convey("Given Metal transposed convolution", test, func() {
		convolutionOps, err := NewConvolutionOps(lib)
		So(err, ShouldBeNil)

		Convey("It should initialize bias and scatter-add on Metal at contract sizes", func() {
			for _, outputElements := range []int{1, 7, 64, 1024, 8192} {
				input := metalConvolutionSequence(outputElements, 0.017, -0.11)
				weight := []float64{0.375}
				bias := []float64{-0.25}
				expected := metalReferenceConvTranspose2d(
					input,
					weight,
					bias,
					1,
					1,
					1,
					outputElements,
					1,
					1,
					1,
					1,
					1,
					1,
				)

				actual, err := convolutionOps.ConvTranspose2d(
					input,
					1,
					1,
					1,
					outputElements,
					weight,
					bias,
					1,
					1,
					1,
					1,
					1,
					0,
					0,
					1,
					1,
					1,
					1,
					outputElements,
				)

				So(err, ShouldBeNil)
				assertMetalConvolutionSlice(
					fmt.Sprintf("conv_transpose2d/%d", outputElements),
					actual,
					expected,
					1e-6,
				)
			}
		})
	})
}

func BenchmarkConvolutionOps_ConvTranspose2d(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "convolution.metallib")
	convolutionOps, err := NewConvolutionOps(lib)

	if err != nil {
		benchmark.Fatal(err)
	}

	input := metalConvolutionSequence(8192, 0.017, -0.11)
	weight := []float64{0.375}
	bias := []float64{-0.25}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := convolutionOps.ConvTranspose2d(
			input,
			1,
			1,
			1,
			8192,
			weight,
			bias,
			1,
			1,
			1,
			1,
			1,
			0,
			0,
			1,
			1,
			1,
			1,
			8192,
		); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func metalConvolutionSequence(length int, scale float64, offset float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		values[index] = offset + scale*float64((index%17)-8)
	}

	return values
}

func metalReferenceConvTranspose2d(
	input []float64,
	weight []float64,
	bias []float64,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	groups int,
) []float64 {
	heightOut := (height-1)*strideHeight + kernelHeight
	widthOut := (width-1)*strideWidth + kernelWidth
	output := make([]float64, batch*outChannels*heightOut*widthOut)
	inChannelsPerGroup := inChannels / groups
	outChannelsPerGroup := outChannels / groups

	for batchIndex := range batch {
		for outputChannel := range outChannels {
			base := (batchIndex*outChannels + outputChannel) * heightOut * widthOut

			for outputIndex := range heightOut * widthOut {
				output[base+outputIndex] = bias[outputChannel]
			}
		}
	}

	for batchIndex := range batch {
		for groupIndex := range groups {
			inputChannelStart := groupIndex * inChannelsPerGroup
			outputChannelStart := groupIndex * outChannelsPerGroup

			for inputChannelOffset := range inChannelsPerGroup {
				inputChannel := inputChannelStart + inputChannelOffset
				weightBase := inputChannel * outChannelsPerGroup * kernelHeight * kernelWidth

				for inputRow := range height {
					for inputColumn := range width {
						inputOffset := ((batchIndex*inChannels+inputChannel)*height+inputRow)*width + inputColumn
						inputValue := input[inputOffset]

						for outputChannelOffset := range outChannelsPerGroup {
							outputChannel := outputChannelStart + outputChannelOffset

							for kernelRow := range kernelHeight {
								outputRow := inputRow*strideHeight + kernelRow

								for kernelColumn := range kernelWidth {
									outputColumn := inputColumn*strideWidth + kernelColumn
									weightOffset := weightBase + (outputChannelOffset*kernelHeight+kernelRow)*kernelWidth + kernelColumn
									outputOffset := ((batchIndex*outChannels+outputChannel)*heightOut+outputRow)*widthOut + outputColumn
									output[outputOffset] += inputValue * weight[weightOffset]
								}
							}
						}
					}
				}
			}
		}
	}

	return output
}

func assertMetalConvolutionSlice(
	name string,
	actual []float64,
	expected []float64,
	tolerance float64,
) {
	So(actual, ShouldHaveLength, len(expected))

	maxDiff := 0.0
	maxIndex := 0

	for index := range expected {
		diff := actual[index] - expected[index]

		if diff < 0 {
			diff = -diff
		}

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
