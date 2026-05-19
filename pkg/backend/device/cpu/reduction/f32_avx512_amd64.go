//go:build amd64

package reduction

//go:noescape
func SumFloat32AVX512Asm(values *float32, count int) float32

//go:noescape
func ProdFloat32AVX512Asm(values *float32, count int) float32

//go:noescape
func ReduceMaxFloat32AVX512Asm(values *float32, count int) float32

//go:noescape
func ReduceMinFloat32AVX512Asm(values *float32, count int) float32

//go:noescape
func L1NormFloat32AVX512Asm(values *float32, count int) float32

/*
SumF32AVX512 accumulates in f64 (scalar reference contract) then narrows once.
*/
func SumF32AVX512(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return SumFloat32AVX512Asm(values, count)
}

/*
ProdF32AVX512 multiplies in f32 vector lanes then folds (NEON-equivalent order).
*/
func ProdF32AVX512(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdFloat32AVX512Asm(values, count)
}

/*
MaxF32AVX512 reduces with per-lane VMAXPS then horizontal fold.
*/
func MaxF32AVX512(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ReduceMaxFloat32AVX512Asm(values, count)
}

/*
MinF32AVX512 reduces with per-lane VMINPS then horizontal fold.
*/
func MinF32AVX512(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return ReduceMinFloat32AVX512Asm(values, count)
}

/*
L1NormF32AVX512 sums |x| in f64 after masked abs on vector lanes.
*/
func L1NormF32AVX512(values *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormFloat32AVX512Asm(values, count)
}
