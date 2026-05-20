//go:build amd64

package active_inference

import "unsafe"

//go:noescape
func PrecisionWeightFloat32AVX512Asm(errors, precision, output *float32, count int)

//go:noescape
func BeliefUpdateFloat32AVX512Asm(likelihood, prior, output *float32, count int)

//go:noescape
func FreeEnergyFloat32AVX512Asm(likelihood, posterior, prior *float32, count int) float32

//go:noescape
func ExpectedFreeEnergyFloat32AVX512Asm(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32

func PrecisionWeightF32AVX512(errors, precision, output *float32, count int) {
	if count == 0 {
		return
	}

	PrecisionWeightFloat32AVX512Asm(errors, precision, output, count)
}

func BeliefUpdateF32AVX512(likelihood, prior, output *float32, count int) {
	if count == 0 {
		return
	}

	BeliefUpdateFloat32AVX512Asm(likelihood, prior, output, count)
}

func FreeEnergyF32AVX512(likelihood, posterior, prior *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	blockCount := count &^ 7
	var total float64

	if blockCount > 0 {
		total += float64(FreeEnergyFloat32AVX512Asm(likelihood, posterior, prior, blockCount))
	}

	tailCount := count - blockCount

	if tailCount == 0 {
		return float32(total)
	}

	likeView := unsafe.Slice(likelihood, count)[blockCount:]
	postView := unsafe.Slice(posterior, count)[blockCount:]
	priorView := unsafe.Slice(prior, count)[blockCount:]

	total += float64(FreeEnergyFloat32Scalar(likeView, postView, priorView))

	return float32(total)
}

func ExpectedFreeEnergyF32AVX512(
	predictedObs, preferredObs, predictedState *float32,
	obsCount, stateCount int,
) float32 {
	if obsCount == 0 {
		return 0
	}

	obsBlock := obsCount &^ 7
	stateBlock := stateCount &^ 7
	var total float64

	if obsBlock > 0 || stateBlock > 0 {
		total += float64(ExpectedFreeEnergyFloat32AVX512Asm(
			predictedObs, preferredObs, predictedState,
			obsBlock, stateBlock,
		))
	}

	if obsBlock < obsCount {
		obsTail := unsafe.Slice(predictedObs, obsCount)[obsBlock:]
		prefTail := unsafe.Slice(preferredObs, obsCount)[obsBlock:]

		total += float64(pragmaticTermFloat32Scalar(obsTail, prefTail))
	}

	if stateBlock < stateCount {
		stateTail := unsafe.Slice(predictedState, stateCount)[stateBlock:]

		total += float64(epistemicTermFloat32Scalar(stateTail))
	}

	return float32(total)
}
