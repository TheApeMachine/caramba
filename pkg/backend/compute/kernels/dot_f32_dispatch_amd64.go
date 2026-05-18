//go:build amd64

package kernels

/*
amd64 dispatcher for f32 dot product. AVX-512 / AVX2 / SSE2 paths
(VFMADD231PS, etc.) land in .s files in a hardware-verified session.
*/

func dotFloat32Native(a, b []float32) float32 {
	var sum float64

	for index := range a {
		sum += float64(a[index]) * float64(b[index])
	}

	return float32(sum)
}
