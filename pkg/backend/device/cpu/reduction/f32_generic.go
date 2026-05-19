package reduction

import (
	"math"
	"unsafe"
)

func SumF32Generic(values *float32, count int) float32 {
	view := unsafe.Slice(values, count)
	var sum float64

	for index := 0; index < count; index++ {
		sum += float64(view[index])
	}

	return float32(sum)
}

func ProdF32Generic(values *float32, count int) float32 {
	view := unsafe.Slice(values, count)
	product := float64(1)

	for index := 0; index < count; index++ {
		product *= float64(view[index])
	}

	return float32(product)
}

func MinF32Generic(values *float32, count int) float32 {
	view := unsafe.Slice(values, count)
	minimum := view[0]

	for index := 1; index < count; index++ {
		if view[index] < minimum {
			minimum = view[index]
		}
	}

	return minimum
}

func MaxF32Generic(values *float32, count int) float32 {
	view := unsafe.Slice(values, count)
	maximum := view[0]

	for index := 1; index < count; index++ {
		if view[index] > maximum {
			maximum = view[index]
		}
	}

	return maximum
}

func L1NormF32Generic(values *float32, count int) float32 {
	view := unsafe.Slice(values, count)
	var sum float64

	for index := 0; index < count; index++ {
		sum += math.Abs(float64(view[index]))
	}

	return float32(sum)
}
