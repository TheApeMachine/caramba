package dot

import "unsafe"

func DotF32Generic(left, right *float32, count int) float32 {
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)
	var sum float64

	for index := 0; index < count; index++ {
		sum += float64(leftView[index]) * float64(rightView[index])
	}

	return float32(sum)
}
