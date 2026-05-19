//go:build arm64

package cpu

func SumFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return SumFloat32NEONAsm(&values[0], len(values))
}
