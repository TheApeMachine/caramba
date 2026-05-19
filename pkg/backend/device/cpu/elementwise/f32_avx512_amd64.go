//go:build amd64

package elementwise

//go:noescape
func AddFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func SubFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func MulFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func DivFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func MaxFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func MinFloat32AVX512Asm(dst, left, right *float32, count int)

//go:noescape
func AbsFloat32AVX512Asm(dst, src *float32, count int)

//go:noescape
func NegFloat32AVX512Asm(dst, src *float32, count int)

//go:noescape
func SqrtFloat32AVX512Asm(dst, src *float32, count int)

//go:noescape
func ReluFloat32AVX512Asm(dst, src *float32, count int)

//go:noescape
func AxpyFloat32AVX512Asm(y, x *float32, alpha float32, count int)

func AddF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	AddFloat32AVX512Asm(dst, left, right, count)
}

func SubF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	SubFloat32AVX512Asm(dst, left, right, count)
}

func MulF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	MulFloat32AVX512Asm(dst, left, right, count)
}

func DivF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	DivFloat32AVX512Asm(dst, left, right, count)
}

func MaxF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	MaxFloat32AVX512Asm(dst, left, right, count)
}

func MinF32AVX512(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	MinFloat32AVX512Asm(dst, left, right, count)
}

func AbsF32AVX512(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	AbsFloat32AVX512Asm(dst, src, count)
}

func NegF32AVX512(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	NegFloat32AVX512Asm(dst, src, count)
}

func SqrtF32AVX512(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	SqrtFloat32AVX512Asm(dst, src, count)
}

func ReluF32AVX512(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	ReluFloat32AVX512Asm(dst, src, count)
}

func AxpyF32AVX512(y, x *float32, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyFloat32AVX512Asm(y, x, alpha, count)
}
