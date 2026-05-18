//go:build arm64

package kernels

/*
NEON entry points for elementwise float32 ops. The assembly body lives
in elementwise_f32_neon_arm64.s and processes 16 lanes per inner
iteration via four 128-bit registers, with a 4-lane secondary loop and
a scalar tail.
*/

//go:noescape
func addFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func subFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func mulFloat32NEONAsm(dst, left, right *float32, n int)

//go:noescape
func divFloat32NEONAsm(dst, left, right *float32, n int)
