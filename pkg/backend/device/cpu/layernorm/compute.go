package layernorm

import (
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

const layerNormEpsilon = 1e-5
const rmsNormEpsilon = 1e-6

func dispatchLayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	elementCount := rows * lastDim

	switch format {
	case dtype.Float32:
		runLayerNormF32(input, scale, bias, output, rows, lastDim)
	case dtype.Float16, dtype.BFloat16:
		inputF32 := widenBuffer(input, elementCount, format)
		scaleF32 := widenBuffer(scale, lastDim, format)
		biasF32 := widenBuffer(bias, lastDim, format)
		outF32 := BorrowFloat32Buffer(elementCount)

		defer ReleaseFloat32Buffer(inputF32)
		defer ReleaseFloat32Buffer(scaleF32)
		defer ReleaseFloat32Buffer(biasF32)
		defer ReleaseFloat32Buffer(outF32)

		runLayerNormF32(
			unsafe.Pointer(&inputF32[0]),
			unsafe.Pointer(&scaleF32[0]),
			unsafe.Pointer(&biasF32[0]),
			unsafe.Pointer(&outF32[0]),
			rows,
			lastDim,
		)

		narrowBuffer(output, outF32, format)
	default:
		panic("layernorm: unsupported dtype")
	}
}

func dispatchRMSNorm(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	elementCount := rows * lastDim

	switch format {
	case dtype.Float32:
		runRMSNormF32(input, scale, output, rows, lastDim)
	case dtype.Float16, dtype.BFloat16:
		inputF32 := widenBuffer(input, elementCount, format)
		scaleF32 := widenBuffer(scale, lastDim, format)
		outF32 := BorrowFloat32Buffer(elementCount)

		defer ReleaseFloat32Buffer(inputF32)
		defer ReleaseFloat32Buffer(scaleF32)
		defer ReleaseFloat32Buffer(outF32)

		runRMSNormF32(
			unsafe.Pointer(&inputF32[0]),
			unsafe.Pointer(&scaleF32[0]),
			unsafe.Pointer(&outF32[0]),
			rows,
			lastDim,
		)

		narrowBuffer(output, outF32, format)
	default:
		panic("layernorm: unsupported dtype")
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
		panic("layernorm: unsupported dtype")
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
		panic("layernorm: unsupported dtype")
	}
}

func runLayerNormF32(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*float32)(input), rows*lastDim)
	scaleView := unsafe.Slice((*float32)(scale), lastDim)
	biasView := unsafe.Slice((*float32)(bias), lastDim)
	outputView := unsafe.Slice((*float32)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]

		mean := computeRowMean(row)
		variance := computeRowVariance(row, mean)
		invStdDev := 1.0 / math.Sqrt(variance+layerNormEpsilon)
		applyRowNormalization(row, outRow, scaleView, biasView, mean, invStdDev)
	}
}

func runRMSNormF32(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
) {
	inputView := unsafe.Slice((*float32)(input), rows*lastDim)
	scaleView := unsafe.Slice((*float32)(scale), lastDim)
	outputView := unsafe.Slice((*float32)(output), rows*lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := inputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := outputView[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRow(row, outRow, scaleView)
	}
}

func computeRowMean(row []float32) float64 {
	return float64(SumFloat32Native(row)) / float64(len(row))
}

func computeRowVariance(row []float32, mean float64) float64 {
	return float64(LayerNormSquaredDiffSumNative(row, float32(mean))) / float64(len(row))
}

func applyRowNormalization(
	row, outRow, scale, bias []float32,
	mean, invStdDev float64,
) {
	LayerNormApplyRowNative(outRow, row, scale, bias, float32(mean), float32(invStdDev))
}

func applyRMSRow(row, outRow, scale []float32) {
	sumOfSquares := DotFloat32Native(row, row)
	meanSquare := float64(sumOfSquares) / float64(len(row))
	invRMS := 1.0 / math.Sqrt(meanSquare+rmsNormEpsilon)
	invRMSf32 := float32(invRMS)

	combined := BorrowFloat32Buffer(len(row))
	defer ReleaseFloat32Buffer(combined)

	for index := range scale {
		combined[index] = invRMSf32 * scale[index]
	}

	MulFloat32Native(outRow, row, combined)
}
