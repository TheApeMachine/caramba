package masking

import (
	"math"
	"unsafe"
)

func ApplyMask(input, mask, output unsafe.Pointer, count int) {
	if count == 0 {
		return
	}

	inputView := unsafe.Slice((*float32)(input), count)
	maskView := unsafe.Slice((*float32)(mask), count)
	outputView := unsafe.Slice((*float32)(output), count)

	for index, value := range inputView {
		outputView[index] = value + maskView[index]
	}
}

func CausalMask(output unsafe.Pointer, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	outputView := unsafe.Slice((*float32)(output), seqQ*seqK)

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for colIndex := 0; colIndex < seqK; colIndex++ {
			if colIndex > rowIndex {
				outputView[rowIndex*seqK+colIndex] = float32(math.Inf(-1))
				continue
			}

			outputView[rowIndex*seqK+colIndex] = 0
		}
	}
}

func ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	scoresView := unsafe.Slice((*float32)(scores), seqQ*seqK)
	slopeView := unsafe.Slice((*float32)(slope), 1)
	outputView := unsafe.Slice((*float32)(output), seqQ*seqK)
	slopeValue := slopeView[0]

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex

			if distance < 0 {
				outputView[index] = scoresView[index]
				continue
			}

			outputView[index] = scoresView[index] - slopeValue*float32(distance)
		}
	}
}
