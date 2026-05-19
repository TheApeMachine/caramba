package dot

import "unsafe"

func DotInt8Generic(left, right *int8, count int) int32 {
	leftView := unsafe.Slice(left, count)
	rightView := unsafe.Slice(right, count)
	var sum int32

	for index := 0; index < count; index++ {
		sum += int32(leftView[index]) * int32(rightView[index])
	}

	return sum
}
