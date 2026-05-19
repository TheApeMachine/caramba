//go:build !arm64

package cpu

func DotInt8Native(a, b []int8) int32 {
	var sum int32

	for index := range a {
		sum += int32(a[index]) * int32(b[index])
	}

	return sum
}
