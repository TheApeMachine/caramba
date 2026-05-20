package rope

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchRoPE(
	config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	if seqLen == 0 || numHeads == 0 || headDim == 0 || headDim%2 != 0 {
		return
	}

	elementCount := seqLen * numHeads * headDim

	switch format {
	case dtype.Float32:
		runRoPEF32(config, input, output, seqLen, numHeads, headDim)
	case dtype.Float16, dtype.BFloat16:
		inputF32 := widenBuffer(input, elementCount, format)
		outputF32 := BorrowFloat32Buffer(elementCount)

		defer ReleaseFloat32Buffer(inputF32)
		defer ReleaseFloat32Buffer(outputF32)

		runRoPEF32(
			config,
			unsafe.Pointer(&inputF32[0]),
			unsafe.Pointer(&outputF32[0]),
			seqLen,
			numHeads,
			headDim,
		)

		narrowBuffer(output, outputF32, format)
	default:
		panic("rope: unsupported dtype")
	}
}

func widenBuffer(source unsafe.Pointer, count int, format dtype.DType) []float32 {
	buffer := BorrowFloat32Buffer(count)

	switch format {
	case dtype.Float32:
		sourceView := unsafe.Slice((*float32)(source), count)
		copy(buffer, sourceView)
	case dtype.Float16:
		sourceView := unsafe.Slice((*dtype.F16)(source), count)
		Float16BulkToFloat32(buffer, sourceView)
	case dtype.BFloat16:
		sourceView := unsafe.Slice((*dtype.BF16)(source), count)
		Bfloat16BulkToFloat32(buffer, sourceView)
	default:
		panic("rope: unsupported dtype")
	}

	return buffer
}

func narrowBuffer(destination unsafe.Pointer, source []float32, format dtype.DType) {
	switch format {
	case dtype.Float32:
		destinationView := unsafe.Slice((*float32)(destination), len(source))
		copy(destinationView, source)
	case dtype.Float16:
		destinationView := unsafe.Slice((*dtype.F16)(destination), len(source))
		Float32BulkToFloat16(destinationView, source)
	case dtype.BFloat16:
		destinationView := unsafe.Slice((*dtype.BF16)(destination), len(source))
		Float32BulkToBFloat16(destinationView, source)
	default:
		panic("rope: unsupported dtype")
	}
}

func runRoPEF32(
	config RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
) {
	inputView := unsafe.Slice((*float32)(input), seqLen*numHeads*headDim)
	outputView := unsafe.Slice((*float32)(output), seqLen*numHeads*headDim)
	halfDim := headDim / 2

	cosBuffer := BorrowFloat32Buffer(halfDim)
	sinBuffer := BorrowFloat32Buffer(halfDim)

	defer ReleaseFloat32Buffer(cosBuffer)
	defer ReleaseFloat32Buffer(sinBuffer)

	for seqIndex := 0; seqIndex < seqLen; seqIndex++ {
		position := float64(seqIndex + config.StartPosition)

		for pairIndex := 0; pairIndex < halfDim; pairIndex++ {
			exponent := -float64(2*pairIndex) / float64(headDim)
			theta := position * math.Pow(config.BaseFreq, exponent)
			cosBuffer[pairIndex] = float32(math.Cos(theta))
			sinBuffer[pairIndex] = float32(math.Sin(theta))
		}

		for headIndex := 0; headIndex < numHeads; headIndex++ {
			rowOffset := (seqIndex*numHeads + headIndex) * headDim
			RopePairsNative(
				outputView[rowOffset:rowOffset+headDim],
				inputView[rowOffset:rowOffset+headDim],
				cosBuffer,
				sinBuffer,
			)
		}
	}
}
