package losses

import "unsafe"

func runMSEF32(predictions, targets unsafe.Pointer, count int) float32 {
	sum := mseSumF32Kernel((*float32)(predictions), (*float32)(targets), count)
	return sum / float32(count)
}

func runMAEF32(predictions, targets unsafe.Pointer, count int) float32 {
	sum := maeSumF32Kernel((*float32)(predictions), (*float32)(targets), count)
	return sum / float32(count)
}

func runHuberF32(predictions, targets unsafe.Pointer, count int) float32 {
	return huberMeanF32(predictions, targets, count)
}

func runBinaryCrossEntropyF32(predictions, targets unsafe.Pointer, count int) float32 {
	return binaryCrossEntropyMeanF32(predictions, targets, count)
}

func runKLDivergenceF32(predictions, targets unsafe.Pointer, count int) float32 {
	return klDivergenceMeanF32(predictions, targets, count)
}
