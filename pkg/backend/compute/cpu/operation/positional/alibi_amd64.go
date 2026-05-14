//go:build amd64

package positional

//go:noescape
func ALiBiRowAVX2(dst []float64, slope float64, q, seqLenK int)

//go:noescape
func ALiBiRowSSE2(dst []float64, slope float64, q, seqLenK int)

func alibiKernel(out, slopes []float64, seqLenQ, seqLenK int, causal bool) {
	numHeads := len(slopes)

	for headIndex := 0; headIndex < numHeads; headIndex++ {
		slope := slopes[headIndex]

		for queryIndex := 0; queryIndex < seqLenQ; queryIndex++ {
			offset := (headIndex*seqLenQ + queryIndex) * seqLenK
			row := out[offset : offset+seqLenK]

			if useAVX2 {
				ALiBiRowAVX2(row, slope, queryIndex, seqLenK)
			} else {
				ALiBiRowSSE2(row, slope, queryIndex, seqLenK)
			}

			if !causal {
				for keyIndex := 0; keyIndex < seqLenK; keyIndex++ {
					if row[keyIndex] < 0 {
						row[keyIndex] = -row[keyIndex]
					}
				}
			}
		}
	}
}
