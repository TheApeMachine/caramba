package reduction

import "unsafe"

func runSumF32(values unsafe.Pointer, count int) float32 {
	return sumF32Kernel((*float32)(values), count)
}

func runProdF32(values unsafe.Pointer, count int) float32 {
	return prodF32Kernel((*float32)(values), count)
}

func runMinF32(values unsafe.Pointer, count int) float32 {
	return minF32Kernel((*float32)(values), count)
}

func runMaxF32(values unsafe.Pointer, count int) float32 {
	return maxF32Kernel((*float32)(values), count)
}

func runL1NormF32(values unsafe.Pointer, count int) float32 {
	return l1NormF32Kernel((*float32)(values), count)
}
