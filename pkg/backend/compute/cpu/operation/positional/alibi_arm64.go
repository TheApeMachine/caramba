//go:build arm64

package positional

//go:noescape
func ALiBiRowNEON(dst []float64, slope float64, q, seqLenK int)

func applyALiBi(out, slopes []float64, seqLenQ, seqLenK int, causal bool) {
	numHeads := len(slopes)
	for h := 0; h < numHeads; h++ {
		for q := 0; q < seqLenQ; q++ {
			offset := (h*seqLenQ + q) * seqLenK
			row := out[offset : offset+seqLenK]
			ALiBiRowNEON(row, slopes[h], q, seqLenK)
			if !causal {
				for k := 0; k < seqLenK; k++ {
					if row[k] < 0 {
						row[k] = -row[k]
					}
				}
			}
		}
	}
}
