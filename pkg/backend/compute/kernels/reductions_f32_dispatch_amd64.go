//go:build amd64

package kernels

/*
amd64 dispatcher for f32 sum reduction. AVX-512 / AVX2 / SSE2 land in
a hardware-verified session; today this routes through the scalar
reference which uses f64 accumulation.
*/

func sumFloat32Native(values []float32) float32 {
	var sum float64

	for _, value := range values {
		sum += float64(value)
	}

	return float32(sum)
}
