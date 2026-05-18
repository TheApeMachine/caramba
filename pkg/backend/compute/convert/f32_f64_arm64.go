//go:build arm64

package convert

/*
ARM64 NEON paths for F32↔F64 using FCVTL / FCVTN against the 2D / 2S
type pair (single instruction widens 2 f32 → 2 f64 or narrows back).
*/

//go:noescape
func float32ToFloat64NEONAsm(dst *float64, src *float32, n int) int

//go:noescape
func float64ToFloat32NEONAsm(dst *float32, src *float64, n int) int

func float32ToFloat64Native(dst []float64, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	if len(src) == 0 {
		return nil
	}

	float32ToFloat64NEONAsm(&dst[0], &src[0], len(src))
	return nil
}

func float64ToFloat32Native(dst []float32, src []float64) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	if len(src) == 0 {
		return nil
	}

	float64ToFloat32NEONAsm(&dst[0], &src[0], len(src))
	return nil
}
