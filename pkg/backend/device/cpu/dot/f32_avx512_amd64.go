//go:build amd64

package dot

//go:noescape
func DotFloat32AVX512Asm(left, right *float32, count int) float32

func DotF32AVX512(left, right *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return DotFloat32AVX512Asm(left, right, count)
}
