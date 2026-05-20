package normalization

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

const normEpsilon = 1e-5

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

func widenBuffer(source unsafe.Pointer, count int, format dtype.DType) []float32 {
	buffer := BorrowFloat32Buffer(count)

	switch format {
	case dtype.Float32:
		sourceView := unsafe.Slice((*float32)(source), count)
		copy(buffer, sourceView)
	case dtype.Float16:
		for index := 0; index < count; index++ {
			buffer[index] = loadF16(source, index)
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			buffer[index] = loadBF16(source, index)
		}
	default:
		panic("normalization: unsupported dtype")
	}

	return buffer
}

func narrowBuffer(destination unsafe.Pointer, source []float32, format dtype.DType) {
	switch format {
	case dtype.Float32:
		destinationView := unsafe.Slice((*float32)(destination), len(source))
		copy(destinationView, source)
	case dtype.Float16:
		for index, value := range source {
			storeF16(destination, index, value)
		}
	case dtype.BFloat16:
		for index, value := range source {
			storeBF16(destination, index, value)
		}
	default:
		panic("normalization: unsupported dtype")
	}
}

func dispatchGroupNorm(
	config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	if channels%config.Groups != 0 {
		panic("normalization: channels must divide groups evenly")
	}

	elementCount := batch * channels * spatial

	switch format {
	case dtype.Float32:
		runGroupNormF32(config, input, scale, bias, output, batch, channels, spatial)
	case dtype.Float16, dtype.BFloat16:
		inputF32 := widenBuffer(input, elementCount, format)
		scaleF32 := widenBuffer(scale, channels, format)
		biasF32 := widenBuffer(bias, channels, format)
		outF32 := BorrowFloat32Buffer(elementCount)

		defer ReleaseFloat32Buffer(inputF32)
		defer ReleaseFloat32Buffer(scaleF32)
		defer ReleaseFloat32Buffer(biasF32)
		defer ReleaseFloat32Buffer(outF32)

		groupNormSlices(
			config,
			inputF32,
			scaleF32,
			biasF32,
			outF32,
			batch,
			channels,
			spatial,
		)

		narrowBuffer(output, outF32, format)
	default:
		panic("normalization: unsupported dtype")
	}
}

func dispatchInstanceNorm(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	elementCount := batch * channels * spatial

	switch format {
	case dtype.Float32:
		runInstanceNormF32(input, scale, bias, output, batch, channels, spatial)
	case dtype.Float16, dtype.BFloat16:
		inputF32 := widenBuffer(input, elementCount, format)
		scaleF32 := widenBuffer(scale, channels, format)
		biasF32 := widenBuffer(bias, channels, format)
		outF32 := BorrowFloat32Buffer(elementCount)

		defer ReleaseFloat32Buffer(inputF32)
		defer ReleaseFloat32Buffer(scaleF32)
		defer ReleaseFloat32Buffer(biasF32)
		defer ReleaseFloat32Buffer(outF32)

		instanceNormSlices(inputF32, scaleF32, biasF32, outF32, batch, channels, spatial)
		narrowBuffer(output, outF32, format)
	default:
		panic("normalization: unsupported dtype")
	}
}

func dispatchBatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	elementCount := batch * channels * spatial

	switch format {
	case dtype.Float32:
		runBatchNormEvalF32(input, scale, bias, mean, variance, output, batch, channels, spatial)
	case dtype.Float16, dtype.BFloat16:
		inputF32 := widenBuffer(input, elementCount, format)
		scaleF32 := widenBuffer(scale, channels, format)
		biasF32 := widenBuffer(bias, channels, format)
		meanF32 := widenBuffer(mean, channels, format)
		varianceF32 := widenBuffer(variance, channels, format)
		outF32 := BorrowFloat32Buffer(elementCount)

		defer ReleaseFloat32Buffer(inputF32)
		defer ReleaseFloat32Buffer(scaleF32)
		defer ReleaseFloat32Buffer(biasF32)
		defer ReleaseFloat32Buffer(meanF32)
		defer ReleaseFloat32Buffer(varianceF32)
		defer ReleaseFloat32Buffer(outF32)

		batchNormEvalSlices(
			inputF32, scaleF32, biasF32, meanF32, varianceF32, outF32,
			batch, channels, spatial,
		)

		narrowBuffer(output, outF32, format)
	default:
		panic("normalization: unsupported dtype")
	}
}

func runGroupNormF32(
	config GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*float32)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*float32)(scale), channels)
	biasView := unsafe.Slice((*float32)(bias), channels)
	outputView := unsafe.Slice((*float32)(output), batch*channels*spatial)

	groupNormSlices(config, inputView, scaleView, biasView, outputView, batch, channels, spatial)
}

func runInstanceNormF32(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*float32)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*float32)(scale), channels)
	biasView := unsafe.Slice((*float32)(bias), channels)
	outputView := unsafe.Slice((*float32)(output), batch*channels*spatial)

	instanceNormSlices(inputView, scaleView, biasView, outputView, batch, channels, spatial)
}

func runBatchNormEvalF32(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
) {
	inputView := unsafe.Slice((*float32)(input), batch*channels*spatial)
	scaleView := unsafe.Slice((*float32)(scale), channels)
	biasView := unsafe.Slice((*float32)(bias), channels)
	meanView := unsafe.Slice((*float32)(mean), channels)
	varianceView := unsafe.Slice((*float32)(variance), channels)
	outputView := unsafe.Slice((*float32)(output), batch*channels*spatial)

	batchNormEvalSlices(
		inputView, scaleView, biasView, meanView, varianceView, outputView,
		batch, channels, spatial,
	)
}

func groupNormSlices(
	config GroupNormConfig,
	input, scale, bias, output []float32,
	batch, channels, spatial int,
) {
	channelsPerGroup := channels / config.Groups
	groupSize := channelsPerGroup * spatial

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for groupIndex := 0; groupIndex < config.Groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupStart := batchIndex*channels*spatial + channelStart*spatial

			normalizeGroup(
				input[groupStart:groupStart+groupSize],
				output[groupStart:groupStart+groupSize],
				scale[channelStart:channelStart+channelsPerGroup],
				bias[channelStart:channelStart+channelsPerGroup],
				channelsPerGroup,
				spatial,
			)
		}
	}
}

func normalizeGroup(
	inputSlice, outSlice, scaleSlice, biasSlice []float32,
	channelsPerGroup, spatial int,
) {
	elementCount := len(inputSlice)
	mean := float64(SumFloat32Native(inputSlice)) / float64(elementCount)
	variance := float64(NormSquaredDiffSumNative(inputSlice, float32(mean))) / float64(elementCount)
	invStdDev := 1.0 / math.Sqrt(variance+normEpsilon)
	meanF32 := float32(mean)
	invStdDevF32 := float32(invStdDev)

	for channelIndex := 0; channelIndex < channelsPerGroup; channelIndex++ {
		channelStart := channelIndex * spatial
		row := inputSlice[channelStart : channelStart+spatial]
		outRow := outSlice[channelStart : channelStart+spatial]

		NormApplyConstScaleBiasNative(
			outRow, row,
			meanF32, invStdDevF32,
			scaleSlice[channelIndex], biasSlice[channelIndex],
		)
	}
}

func instanceNormSlices(input, scale, bias, output []float32, batch, channels, spatial int) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]

			mean := float64(SumFloat32Native(row)) / float64(spatial)
			variance := float64(NormSquaredDiffSumNative(row, float32(mean))) / float64(spatial)
			invStdDev := 1.0 / math.Sqrt(variance+normEpsilon)

			NormApplyConstScaleBiasNative(
				outRow, row,
				float32(mean), float32(invStdDev),
				scale[channelIndex], bias[channelIndex],
			)
		}
	}
}

func batchNormEvalSlices(
	input, scale, bias, mean, variance, output []float32,
	batch, channels, spatial int,
) {
	for channelIndex := 0; channelIndex < channels; channelIndex++ {
		invStdDev := 1.0 / float32(math.Sqrt(float64(variance[channelIndex])+normEpsilon))
		channelMean := mean[channelIndex]
		channelScale := scale[channelIndex]
		channelBias := bias[channelIndex]

		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			start := (batchIndex*channels + channelIndex) * spatial
			row := input[start : start+spatial]
			outRow := output[start : start+spatial]

			NormApplyConstScaleBiasNative(
				outRow, row,
				channelMean, invStdDev,
				channelScale, channelBias,
			)
		}
	}
}
