package metal

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

type norm3DFixture struct {
	inputBytes    []byte
	scaleBytes    []byte
	biasBytes     []byte
	meanBytes     []byte
	varianceBytes []byte
	groupBytes    []byte
	instanceBytes []byte
	batchBytes    []byte
}

func norm3DFixtureForTest(
	batch int,
	channels int,
	spatial int,
	storageDType dtype.DType,
) norm3DFixture {
	input, scale, bias, mean, variance := norm3DValues(batch, channels, spatial)
	inputBytes := encodeNormValuesAsDType(input, storageDType)
	scaleBytes := encodeNormValuesAsDType(scale, storageDType)
	biasBytes := encodeNormValuesAsDType(bias, storageDType)
	meanBytes := encodeNormValuesAsDType(mean, storageDType)
	varianceBytes := encodeNormValuesAsDType(variance, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	scaleStored := decodeDTypeBytesToFloat32(scaleBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	meanStored := decodeDTypeBytesToFloat32(meanBytes, storageDType)
	varianceStored := decodeDTypeBytesToFloat32(varianceBytes, storageDType)

	return norm3DFixture{
		inputBytes:    inputBytes,
		scaleBytes:    scaleBytes,
		biasBytes:     biasBytes,
		meanBytes:     meanBytes,
		varianceBytes: varianceBytes,
		groupBytes: encodeNormValuesAsDType(
			expectedGroupNormValues(inputStored, scaleStored, biasStored, batch, channels, spatial),
			storageDType,
		),
		instanceBytes: encodeNormValuesAsDType(
			expectedInstanceNormValues(inputStored, scaleStored, biasStored, batch, channels, spatial),
			storageDType,
		),
		batchBytes: encodeNormValuesAsDType(
			expectedBatchNormEvalValues(
				inputStored, scaleStored, biasStored, meanStored, varianceStored, batch, channels, spatial,
			),
			storageDType,
		),
	}
}

func norm3DValues(
	batch int,
	channels int,
	spatial int,
) ([]float32, []float32, []float32, []float32, []float32) {
	input := make([]float32, batch*channels*spatial)
	scale := make([]float32, channels)
	bias := make([]float32, channels)
	mean := make([]float32, channels)
	variance := make([]float32, channels)

	for index := range input {
		input[index] = centeredPowerOfTwoValue(index*13+7, 89, 32)
	}

	for index := range channels {
		scale[index] = 1 + centeredPowerOfTwoValue(index*7+3, 31, 128)
		bias[index] = centeredPowerOfTwoValue(index*11+5, 37, 128)
		mean[index] = centeredPowerOfTwoValue(index*17+1, 43, 64)
		variance[index] = 0.5 + float32(index%17)/16
	}

	return input, scale, bias, mean, variance
}

func expectedGroupNormValues(
	input []float32,
	scale []float32,
	bias []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	out := make([]float32, len(input))
	groups := metalDefaultGroupNormGroups
	channelsPerGroup := channels / groups

	for batchIndex := range batch {
		for groupIndex := range groups {
			channelStart := groupIndex * channelsPerGroup
			groupStart := (batchIndex*channels + channelStart) * spatial
			groupSize := channelsPerGroup * spatial
			applyGroupNormExpected(
				input[groupStart:groupStart+groupSize],
				out[groupStart:groupStart+groupSize],
				scale[channelStart:channelStart+channelsPerGroup],
				bias[channelStart:channelStart+channelsPerGroup],
				channelsPerGroup,
				spatial,
			)
		}
	}

	return out
}

func expectedInstanceNormValues(
	input []float32,
	scale []float32,
	bias []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	out := make([]float32, len(input))

	for batchIndex := range batch {
		for channelIndex := range channels {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := out[start : start+spatial]
			mean := normalizationMean64ForTest(row)
			variance := normalizationVariance64ForTest(row, mean)
			invStdDev := normInvStdDev64(variance)
			applyNorm3DExpectedRow64(row, outRow, scale[channelIndex], bias[channelIndex], mean, invStdDev)
		}
	}

	return out
}

func expectedBatchNormEvalValues(
	input []float32,
	scale []float32,
	bias []float32,
	mean []float32,
	variance []float32,
	batch int,
	channels int,
	spatial int,
) []float32 {
	out := make([]float32, len(input))

	for batchIndex := range batch {
		for channelIndex := range channels {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := out[start : start+spatial]
			invStdDev := normInvStdDev(variance[channelIndex])
			applyNorm3DExpectedRow(
				row, outRow, scale[channelIndex], bias[channelIndex], mean[channelIndex], invStdDev,
			)
		}
	}

	return out
}

func applyGroupNormExpected(
	input []float32,
	out []float32,
	scale []float32,
	bias []float32,
	channelsPerGroup int,
	spatial int,
) {
	mean := normalizationMean64ForTest(input)
	variance := normalizationVariance64ForTest(input, mean)
	invStdDev := normInvStdDev64(variance)

	for channelIndex := range channelsPerGroup {
		start := channelIndex * spatial
		applyNorm3DExpectedRow64(
			input[start:start+spatial],
			out[start:start+spatial],
			scale[channelIndex],
			bias[channelIndex],
			mean,
			invStdDev,
		)
	}
}

func applyNorm3DExpectedRow64(
	input []float32,
	out []float32,
	scale float32,
	bias float32,
	mean float64,
	invStdDev float64,
) {
	for index, value := range input {
		normalized := (float64(value) - mean) * invStdDev
		out[index] = float32(normalized)*scale + bias
	}
}

func applyNorm3DExpectedRow(
	input []float32,
	out []float32,
	scale float32,
	bias float32,
	mean float32,
	invStdDev float32,
) {
	for index, value := range input {
		out[index] = (value-mean)*invStdDev*scale + bias
	}
}

func normInvStdDev(variance float32) float32 {
	return 1 / sqrtFloat32(variance+layerNormEpsilonMetalForTest)
}

func normInvStdDev64(variance float64) float64 {
	return 1 / math.Sqrt(variance+layerNormEpsilonMetalForTest)
}

func normalizationMean64ForTest(row []float32) float64 {
	var sum float64

	for _, value := range row {
		sum += float64(value)
	}

	return sum / float64(len(row))
}

func normalizationVariance64ForTest(row []float32, mean float64) float64 {
	var variance float64

	for _, value := range row {
		delta := float64(value) - mean
		variance += delta * delta
	}

	return variance / float64(len(row))
}

func sqrtFloat32(value float32) float32 {
	return float32(math.Sqrt(float64(value)))
}
