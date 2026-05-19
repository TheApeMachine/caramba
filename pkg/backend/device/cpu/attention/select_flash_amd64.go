//go:build amd64

package attention

import (
	"math"

	"golang.org/x/sys/cpu"
)

func RunFlashAttentionRowNative(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causalMask bool,
) {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := make([]float32, valueDim)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causalMask && keyIndex > rowIndex {
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

		valueRow := valueView[keyIndex*valueDim : (keyIndex+1)*valueDim]
		blockCount := valueDim &^ 15

		if blockCount > 0 && cpu.X86.HasAVX512F {
			flashAttentionOnlineUpdateAVX512(
				&accumulator[0], &valueRow[0],
				alpha, shifted, blockCount,
			)
		}

		for dimIndex := blockCount; dimIndex < valueDim; dimIndex++ {
			accumulator[dimIndex] = accumulator[dimIndex]*alpha + valueRow[dimIndex]*shifted
		}
	}

	outputRow := outView[rowIndex*valueDim : (rowIndex+1)*valueDim]

	if normalizer == 0 {
		for dimIndex := range outputRow {
			outputRow[dimIndex] = 0
		}

		return
	}

	invNormalizer := float32(1) / normalizer
	blockCount := valueDim &^ 15

	if blockCount > 0 && cpu.X86.HasAVX512F {
		flashAttentionScaleAVX512(
			&outputRow[0], &accumulator[0],
			invNormalizer, blockCount,
		)
	}

	for dimIndex := blockCount; dimIndex < valueDim; dimIndex++ {
		outputRow[dimIndex] = accumulator[dimIndex] * invNormalizer
	}
}
