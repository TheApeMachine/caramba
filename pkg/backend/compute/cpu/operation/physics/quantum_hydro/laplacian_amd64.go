//go:build amd64

package quantum_hydro

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func laplacianAxisSetAVX2(out, left, center, right []float64, invH2 float64)

//go:noescape
func laplacianAxisAccAVX2(out, left, center, right []float64, invH2 float64)

//go:noescape
func laplacianAxisSetSSE2(out, left, center, right []float64, invH2 float64)

//go:noescape
func laplacianAxisAccSSE2(out, left, center, right []float64, invH2 float64)

/*
axisSet dispatches the "set" axis primitive (out = (l + r - 2c) * invH2)
to AVX2 when the CPU supports it, otherwise to SSE2. Each ISA kernel lives
in its own .s file with its own vector body; no aliasing between bodies.
*/
func axisSet(out, left, center, right []float64, invH2 float64) {
	if useAVX2 {
		laplacianAxisSetAVX2(out, left, center, right, invH2)

		return
	}

	laplacianAxisSetSSE2(out, left, center, right, invH2)
}

/*
axisAcc dispatches the "accumulate" axis primitive
(out += (l + r - 2c) * invH2).
*/
func axisAcc(out, left, center, right []float64, invH2 float64) {
	if useAVX2 {
		laplacianAxisAccAVX2(out, left, center, right, invH2)

		return
	}

	laplacianAxisAccSSE2(out, left, center, right, invH2)
}

/*
alignedLen returns the largest multiple of the SIMD vector width that
fits within count. Width is 4 doubles for AVX2 and 2 doubles for SSE2.
*/
func alignedLen(count int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return count / width * width
}
