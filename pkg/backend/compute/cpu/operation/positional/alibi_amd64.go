//go:build amd64

package positional

func applyALiBi(out, slopes []float64, seqLenQ, seqLenK int, causal bool) {
	numHeads := len(slopes)

	for h := 0; h < numHeads; h++ {
		slope := slopes[h]

		for q := 0; q < seqLenQ; q++ {
			offset := (h*seqLenQ + q) * seqLenK
			row := out[offset : offset+seqLenK]
			for k := 0; k < seqLenK; k++ {
				distance := float64(k - q)

				if !causal && distance < 0 {
					distance = -distance
				}

				row[k] = slope * distance
			}
		}
	}
}
