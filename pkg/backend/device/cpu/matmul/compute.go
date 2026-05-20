package matmul

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

type reducedLoadFunc func(pointer unsafe.Pointer, index int) float32

type reducedStoreFunc func(pointer unsafe.Pointer, index int, value float32)

func dispatchMatmul(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
	f32 func(out, left, right unsafe.Pointer, rows, inner, cols int),
) {
	switch format {
	case dtype.Float32:
		f32(out, left, right, rows, inner, cols)
	case dtype.Float64:
		runMatmulF64(out, left, right, rows, inner, cols)
	case dtype.Float16:
		runMatmulReduced(out, left, right, rows, inner, cols, loadF16, storeF16)
	case dtype.BFloat16:
		runMatmulReduced(out, left, right, rows, inner, cols, loadBF16, storeBF16)
	}
}

func runMatmulReduced(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	load reducedLoadFunc,
	store reducedStoreFunc,
) {
	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for colIndex := 0; colIndex < cols; colIndex++ {
			var sum float32

			for innerIndex := 0; innerIndex < inner; innerIndex++ {
				leftValue := load(left, rowIndex*inner+innerIndex)
				rightValue := load(right, innerIndex*cols+colIndex)
				sum += leftValue * rightValue
			}

			store(out, rowIndex*cols+colIndex, sum)
		}
	}
}
