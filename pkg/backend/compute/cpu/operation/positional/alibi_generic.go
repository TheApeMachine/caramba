//go:build !amd64 && !arm64

package positional

func alibiKernel(out, slopes []float64, seqLenQ, seqLenK int, causal bool) {
	numHeads := len(slopes)

	for headIndex := range numHeads {
		slope := slopes[headIndex]

		for queryIndex := range seqLenQ {
			offset := (headIndex*seqLenQ + queryIndex) * seqLenK
			row := out[offset : offset+seqLenK]

			for keyIndex := range seqLenK {
				distance := float64(keyIndex - queryIndex)

				if !causal && distance < 0 {
					distance = -distance
				}

				row[keyIndex] = slope * distance
			}
		}
	}
}
