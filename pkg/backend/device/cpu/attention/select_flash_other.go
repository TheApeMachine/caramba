//go:build !arm64 && !amd64

package attention

import "math"

func RunFlashAttentionRowNative(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causal bool,
) {
	runFlashAttentionRow(
		queryView, keyView, valueView, outView,
		rowIndex, seqK, depth, valueDim, scale, causal,
	)
}

func runFlashAttentionRow(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causal bool,
) {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := BorrowFloat32Buffer(valueDim)
	scaleScratch := BorrowFloat32Buffer(valueDim)
	valueScratch := BorrowFloat32Buffer(valueDim)

	defer ReleaseFloat32Buffer(accumulator)
	defer ReleaseFloat32Buffer(scaleScratch)
	defer ReleaseFloat32Buffer(valueScratch)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causal && keyIndex > rowIndex {
			continue
		}

		queryRow := queryView[rowIndex*depth : (rowIndex+1)*depth]
		keyRow := keyView[keyIndex*depth : (keyIndex+1)*depth]
		score := DotFloat32Native(queryRow, keyRow) * scale
		oldMax := maxScore

		if score > maxScore {
			maxScore = score
		}

		alpha := flashExpFloat32(oldMax - maxScore)
		shifted := flashExpFloat32(score - maxScore)
		normalizer = normalizer*alpha + shifted

		fillScaleScratch(scaleScratch, alpha, valueDim)
		MulFloat32Native(accumulator, accumulator, scaleScratch)

		valueRow := valueView[keyIndex*valueDim : (keyIndex+1)*valueDim]
		fillScaleScratch(valueScratch, shifted, valueDim)
		MulFloat32Native(valueScratch, valueScratch, valueRow)
		AddFloat32Native(accumulator, accumulator, valueScratch)
	}

	if normalizer == 0 {
		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			outView[rowIndex*valueDim+dimIndex] = 0
		}

		return
	}

	invNormalizer := float32(1) / normalizer
	fillScaleScratch(scaleScratch, invNormalizer, valueDim)
	MulFloat32Native(
		outView[rowIndex*valueDim:(rowIndex+1)*valueDim],
		accumulator,
		scaleScratch,
	)
}

func fillScaleScratch(scratch []float32, value float32, count int) {
	for index := 0; index < count; index++ {
		scratch[index] = value
	}
}
