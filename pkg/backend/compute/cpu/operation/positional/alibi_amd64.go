//go:build amd64

package positional

//go:noescape
func ALiBiRowAVX2(dst []float64, slope float64, q, seqLenK int)

//go:noescape
func ALiBiRowSSE2(dst []float64, slope float64, q, seqLenK int)

func applyALiBi(out, slopes []float64, seqLenQ, seqLenK int, causal bool) {
	numHeads := len(slopes)
	for h := 0; h < numHeads; h++ {
		for q := 0; q < seqLenQ; q++ {
			offset := (h*seqLenQ + q) * seqLenK
			row := out[offset : offset+seqLenK]
			if useAVX2 {
				ALiBiRowAVX2(row, slopes[h], q, seqLenK)
			} else {
				ALiBiRowSSE2(row, slopes[h], q, seqLenK)
			}
			if !causal {
				// convert to abs: flip negative values
				for k := 0; k < seqLenK; k++ {
					if row[k] < 0 {
						row[k] = -row[k]
					}
				}
			}
		}
	}
}
