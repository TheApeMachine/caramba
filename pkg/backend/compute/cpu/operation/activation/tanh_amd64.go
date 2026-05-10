//go:build amd64

package activation

//go:noescape
func TanhAVX2(dst, src []float64)

//go:noescape
func TanhSSE2(dst, src []float64)

func applyTanh(dst, src []float64) {
	if useAVX2 {
		TanhAVX2(dst, src)
	} else {
		TanhSSE2(dst, src)
	}
}
