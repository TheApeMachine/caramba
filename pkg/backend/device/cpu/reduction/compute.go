package reduction

import (
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func dispatchSum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runSumF32(values, count)
	case dtype.BFloat16:
		bf16Result := dispatchSumBF16(values, count)
		return (&bf16Result).Float32()
	case dtype.Float16:
		fp16Result := dispatchSumFP16(values, count)
		return fp16Result.Float32()
	default:
		panic("reduction: Sum unsupported dtype")
	}
}

func dispatchProd(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runProdF32(values, count)
	case dtype.Float16, dtype.BFloat16:
		var product float32 = 1

		for index := 0; index < count; index++ {
			var value float32

			if format == dtype.Float16 {
				value = loadF16(values, index)
			}

			if format == dtype.BFloat16 {
				value = loadBF16(values, index)
			}

			product *= value
		}

		return product
	default:
		panic("reduction: Prod unsupported dtype")
	}
}

func dispatchReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runMinF32(values, count)
	case dtype.Float16, dtype.BFloat16:
		var minimum float32

		for index := range count {
			var value float32

			if format == dtype.Float16 {
				value = loadF16(values, index)
			}

			if format == dtype.BFloat16 {
				value = loadBF16(values, index)
			}

			if index == 0 || value < minimum {
				minimum = value
			}
		}

		return minimum
	default:
		panic("reduction: ReduceMin unsupported dtype")
	}
}

func dispatchReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runMaxF32(values, count)
	case dtype.Float16, dtype.BFloat16:
		var maximum float32

		for index := range count {
			var value float32

			if format == dtype.Float16 {
				value = loadF16(values, index)
			}

			if format == dtype.BFloat16 {
				value = loadBF16(values, index)
			}

			if index == 0 || value > maximum {
				maximum = value
			}
		}

		return maximum
	default:
		panic("reduction: ReduceMax unsupported dtype")
	}
}

func dispatchL1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runL1NormF32(values, count)
	case dtype.Float16, dtype.BFloat16:
		var norm float32

		for index := range count {
			var value float32

			if format == dtype.Float16 {
				value = loadF16(values, index)
			}

			if format == dtype.BFloat16 {
				value = loadBF16(values, index)
			}

			norm += float32(math.Abs(float64(value)))
		}

		return norm
	default:
		panic("reduction: L1Norm unsupported dtype")
	}
}
