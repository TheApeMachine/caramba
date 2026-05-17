//go:build arm64

package quantum_hydro

//go:noescape
func laplacianAxisSetNEON(out, left, center, right []float64, invH2 float64)

//go:noescape
func laplacianAxisAccNEON(out, left, center, right []float64, invH2 float64)

/*
axisSet on arm64 always routes to the NEON kernel; the platform contract
requires NEON on every supported arm64 target.
*/
func axisSet(out, left, center, right []float64, invH2 float64) {
	laplacianAxisSetNEON(out, left, center, right, invH2)
}

/*
axisAcc on arm64 always routes to the NEON kernel.
*/
func axisAcc(out, left, center, right []float64, invH2 float64) {
	laplacianAxisAccNEON(out, left, center, right, invH2)
}

/*
alignedLen returns the largest multiple of the NEON vector width
(2 doubles per v0..v31 register) that fits within count.
*/
func alignedLen(count int) int {
	return count / 2 * 2
}
