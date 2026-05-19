package masking

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		ApplyMaskFloat32Native(input, mask, output, count)
	case dtype.BFloat16:
		inputView := unsafe.Slice((*dtype.BF16)(input), count)
		maskView := unsafe.Slice((*dtype.BF16)(mask), count)
		outputView := unsafe.Slice((*dtype.BF16)(output), count)

		for index := range inputView {
			sum := (&inputView[index]).Float32() + (&maskView[index]).Float32()
			outputView[index] = dtype.NewBfloat16FromFloat32(sum)
		}
	case dtype.Float16:
		inputView := unsafe.Slice((*dtype.F16)(input), count)
		maskView := unsafe.Slice((*dtype.F16)(mask), count)
		outputView := unsafe.Slice((*dtype.F16)(output), count)

		for index := range inputView {
			outputView[index] = dtype.Fromfloat32(
				inputView[index].Float32() + maskView[index].Float32(),
			)
		}
	default:
		panic("masking: ApplyMask unsupported dtype")
	}
}

func CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		CausalMaskFloat32Native(output, seqQ, seqK)
	case dtype.BFloat16:
		outputView := unsafe.Slice((*dtype.BF16)(output), seqQ*seqK)
		const bf16NegInf = dtype.BF16(0xFF80)

		for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
			for colIndex := 0; colIndex < seqK; colIndex++ {
				if colIndex > rowIndex {
					outputView[rowIndex*seqK+colIndex] = bf16NegInf
					continue
				}

				outputView[rowIndex*seqK+colIndex] = 0
			}
		}
	case dtype.Float16:
		outputView := unsafe.Slice((*dtype.F16)(output), seqQ*seqK)
		const fp16NegInf = dtype.F16(0xFC00)

		for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
			for colIndex := 0; colIndex < seqK; colIndex++ {
				if colIndex > rowIndex {
					outputView[rowIndex*seqK+colIndex] = fp16NegInf
					continue
				}

				outputView[rowIndex*seqK+colIndex] = 0
			}
		}
	default:
		panic("masking: CausalMask unsupported dtype")
	}
}

func ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		ALiBiBiasFloat32Native(scores, slope, output, seqQ, seqK)
	case dtype.BFloat16:
		alibiBiasBF16(scores, slope, output, seqQ, seqK)
	case dtype.Float16:
		alibiBiasF16(scores, slope, output, seqQ, seqK)
	default:
		panic("masking: ALiBiBias unsupported dtype")
	}
}

func alibiBiasBF16(scores, slope, output unsafe.Pointer, seqQ, seqK int) {
	scoresView := unsafe.Slice((*dtype.BF16)(scores), seqQ*seqK)
	slopeView := unsafe.Slice((*dtype.BF16)(slope), 1)
	outputView := unsafe.Slice((*dtype.BF16)(output), seqQ*seqK)
	slopeValue := slopeView[0].Float32()

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex
			score := scoresView[index].Float32()

			if distance < 0 {
				outputView[index] = scoresView[index]
				continue
			}

			outputView[index] = dtype.NewBfloat16FromFloat32(score - slopeValue*float32(distance))
		}
	}
}

func alibiBiasF16(scores, slope, output unsafe.Pointer, seqQ, seqK int) {
	scoresView := unsafe.Slice((*dtype.F16)(scores), seqQ*seqK)
	slopeView := unsafe.Slice((*dtype.F16)(slope), 1)
	outputView := unsafe.Slice((*dtype.F16)(output), seqQ*seqK)
	slopeValue := slopeView[0].Float32()

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex

			if distance < 0 {
				outputView[index] = scoresView[index]
				continue
			}

			outputView[index] = dtype.Fromfloat32(
				scoresView[index].Float32() - slopeValue*float32(distance),
			)
		}
	}
}
