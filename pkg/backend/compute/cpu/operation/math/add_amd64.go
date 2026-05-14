//go:build amd64

package math

func addKernel(out, left, right []float64) {
	width := 2
	vectorAdd := addVecSSE2

	if useAVX2 {
		width = 4
		vectorAdd = addVecAVX2
	}

	limit := len(left) / width * width

	if limit > 0 {
		vectorAdd(out[:limit], left[:limit], right[:limit])
	}

	for index := limit; index < len(left); index++ {
		out[index] = left[index] + right[index]
	}
}
