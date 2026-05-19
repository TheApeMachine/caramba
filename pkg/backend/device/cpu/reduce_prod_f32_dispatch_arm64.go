//go:build arm64

package cpu

func ReduceProdFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return ReduceProdFloat32NEONAsm(&values[0], len(values))
}
