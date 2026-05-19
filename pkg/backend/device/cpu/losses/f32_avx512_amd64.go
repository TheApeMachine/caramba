//go:build amd64

package losses

//go:noescape
func MseSumFloat32AVX512Asm(predictions, targets *float32, count int) float32

//go:noescape
func MaeSumFloat32AVX512Asm(predictions, targets *float32, count int) float32

func MseSumF32AVX512(predictions, targets *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return MseSumFloat32AVX512Asm(predictions, targets, count)
}

func MaeSumF32AVX512(predictions, targets *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaeSumFloat32AVX512Asm(predictions, targets, count)
}
