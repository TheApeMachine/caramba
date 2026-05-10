//go:build amd64

package masking

//go:noescape
func ApplyMaskAVX2(dst, scores, mask []float64)

//go:noescape
func ApplyMaskSSE2(dst, scores, mask []float64)

func applyMaskAdd(dst, scores, mask []float64) {
	if useAVX2 {
		ApplyMaskAVX2(dst, scores, mask)
	} else {
		ApplyMaskSSE2(dst, scores, mask)
	}
}
