//go:build amd64

package kernels

/*
amd64 dispatcher for elementwise float32 add/sub/mul. AVX-512 / AVX2 /
SSE2 assembly bodies (vaddps, vsubps, vmulps on zmm/ymm/xmm) land in
.s files in a hardware-verified session; today this routes through the
scalar reference.
*/

func addFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] + right[index]
	}
}

func subFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] - right[index]
	}
}

func mulFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] * right[index]
	}
}

func divFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] / right[index]
	}
}

func maxFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = right[index]

		if left[index] > right[index] {
			dst[index] = left[index]
		}
	}
}

func minFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = right[index]

		if left[index] < right[index] {
			dst[index] = left[index]
		}
	}
}
