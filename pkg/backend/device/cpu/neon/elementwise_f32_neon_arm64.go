//go:build arm64

package neon

/*
NEON entry points for elementwise float32 ops. The assembly body lives
in elementwise_f32_neon_arm64.s and processes 16 lanes per inner
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
