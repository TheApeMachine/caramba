package math

/*
SoftmaxF32 writes a numerically stable softmax of src into dst.
len(dst) and len(src) must be equal and positive.
*/
func SoftmaxF32(dst, src []float32) {
	if len(src) == 0 {
		return
	}

	maxValue := src[0]

	for index := 1; index < len(src); index++ {
		if src[index] > maxValue {
			maxValue = src[index]
		}
	}

	sumExp := float32(0)

	for index, value := range src {
		shifted := FastExp32(value - maxValue)
		dst[index] = shifted
		sumExp += shifted
	}

	invSum := float32(1) / sumExp

	for index := range dst {
		dst[index] *= invSum
	}
}

/*
LogSoftmaxF32 writes a numerically stable log-softmax of src into dst.
*/
func LogSoftmaxF32(dst, src []float32) {
	if len(src) == 0 {
		return
	}

	maxValue := src[0]

	for index := 1; index < len(src); index++ {
		if src[index] > maxValue {
			maxValue = src[index]
		}
	}

	sumExp := float32(0)

	for _, value := range src {
		sumExp += FastExp32(value - maxValue)
	}

	logSum := FastLog32(sumExp)

	for index, value := range src {
		dst[index] = value - maxValue - logSum
	}
}

func Softmax64(dst, src []float64) {
	if len(src) == 0 {
		return
	}

	maxValue := src[0]

	for index := 1; index < len(src); index++ {
		if src[index] > maxValue {
			maxValue = src[index]
		}
	}

	sumExp := float64(0)

	for index, value := range src {
		shifted := FastExp64(value - maxValue)
		dst[index] = shifted
		sumExp += shifted
	}

	invSum := 1 / sumExp

	for index := range dst {
		dst[index] *= invSum
	}
}

func LogSoftmax64(dst, src []float64) {
	if len(src) == 0 {
		return
	}

	maxValue := src[0]

	for index := 1; index < len(src); index++ {
		if src[index] > maxValue {
			maxValue = src[index]
		}
	}

	sumExp := float64(0)

	for _, value := range src {
		sumExp += FastExp64(value - maxValue)
	}

	logSum := FastLog64(sumExp)

	for index, value := range src {
		dst[index] = value - maxValue - logSum
	}
}
