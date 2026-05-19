//go:build arm64

package elementwise

/*
NEON entry points for elementwise float32 binary ops. The assembly
body lives in f32_neon_arm64.s and processes 16 lanes per inner
iteration via four 128-bit registers, with a 4-lane secondary loop and
a scalar tail.
*/

//go:noescape
func AddFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func SubFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func MulFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func DivFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func MaxFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func MinFloat32NEONAsm(dst, left, right *float32, n int)
