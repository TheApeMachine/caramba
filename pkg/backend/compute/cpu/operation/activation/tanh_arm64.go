//go:build arm64

package activation

//go:noescape
func TanhNEON(dst, src []float64)

func applyTanh(dst, src []float64) {
	TanhNEON(dst, src)
}
