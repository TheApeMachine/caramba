//go:build amd64

package interpretability

//go:noescape
func ActivationSteerFloat32AVX512Asm(
	destination, base, direction *float32,
	coefficient float32,
	count int,
)

func ActivationSteerFloat32AVX512(
	destination, base, direction *float32,
	coefficient float32,
	count int,
) {
	if count == 0 {
		return
	}

	ActivationSteerFloat32AVX512Asm(destination, base, direction, coefficient, count)
}
