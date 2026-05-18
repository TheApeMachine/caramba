//go:build !arm64

package kernels

func axpyFloat32Native(dst []float32, src []float32, alpha float32) {
	for index := range dst {
		dst[index] += alpha * src[index]
	}
}
