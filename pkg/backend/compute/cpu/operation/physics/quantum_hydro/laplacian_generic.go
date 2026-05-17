//go:build !amd64 && !arm64

package quantum_hydro

/*
axisSet on architectures without a registered SIMD body falls back to a
straight scalar Go loop. The scalar form is correct and parity-tested as
the reference everywhere; this dispatch path simply keeps the build green
on uncommon targets (mips, riscv64, wasm) where no vector assembly has
been written yet. Adding a real SIMD body for one of these archs is a
named follow-up — when added, it lives in its own .s file with its own
vector kernel, not aliased to anything.
*/
func axisSet(out, left, center, right []float64, invH2 float64) {
	for index := range out {
		out[index] = (left[index] + right[index] - 2.0*center[index]) * invH2
	}
}

/*
axisAcc accumulates the axis contribution into out.
*/
func axisAcc(out, left, center, right []float64, invH2 float64) {
	for index := range out {
		out[index] += (left[index] + right[index] - 2.0*center[index]) * invH2
	}
}

/*
alignedLen on generic targets is a no-op (vector width 1).
*/
func alignedLen(count int) int { return count }
