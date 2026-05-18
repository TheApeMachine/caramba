package metal

import "math"

func fftExpected(realIn []float32, imagIn []float32, inverse bool) ([]float32, []float32) {
	realOut := append([]float32(nil), realIn...)
	imagOut := append([]float32(nil), imagIn...)

	if physicsIsPowerOfTwo(len(realOut)) {
		physicsCooleyTukey(realOut, imagOut, inverse)
	} else {
		physicsNaiveDFT(realOut, imagOut, inverse)
	}

	if inverse {
		scale := float32(1.0 / float64(len(realOut)))

		for index := range realOut {
			realOut[index] *= scale
			imagOut[index] *= scale
		}
	}

	return realOut, imagOut
}

func physicsIsPowerOfTwo(value int) bool {
	return value > 0 && (value&(value-1)) == 0
}

func physicsCooleyTukey(realOut []float32, imagOut []float32, inverse bool) {
	physicsBitReverse(realOut, imagOut)
	sign := -1.0
	if inverse {
		sign = 1.0
	}

	for length := 2; length <= len(realOut); length <<= 1 {
		physicsCooleyTukeyStage(realOut, imagOut, length, sign)
	}
}

func physicsBitReverse(realOut []float32, imagOut []float32) {
	targetIndex := 0
	count := len(realOut)

	for sourceIndex := 1; sourceIndex < count; sourceIndex++ {
		bit := count >> 1
		for ; targetIndex&bit != 0; bit >>= 1 {
			targetIndex ^= bit
		}

		targetIndex ^= bit
		if sourceIndex < targetIndex {
			realOut[sourceIndex], realOut[targetIndex] = realOut[targetIndex], realOut[sourceIndex]
			imagOut[sourceIndex], imagOut[targetIndex] = imagOut[targetIndex], imagOut[sourceIndex]
		}
	}
}

func physicsCooleyTukeyStage(realOut []float32, imagOut []float32, length int, sign float64) {
	angle := sign * 2 * math.Pi / float64(length)
	twiddleStepReal := float32(math.Cos(angle))
	twiddleStepImag := float32(math.Sin(angle))

	for start := 0; start < len(realOut); start += length {
		physicsCooleyTukeyBlock(realOut, imagOut, start, length, twiddleStepReal, twiddleStepImag)
	}
}

func physicsCooleyTukeyBlock(
	realOut []float32,
	imagOut []float32,
	start int,
	length int,
	twiddleStepReal float32,
	twiddleStepImag float32,
) {
	currentReal := float32(1)
	currentImag := float32(0)
	halfLength := length / 2

	for offset := range halfLength {
		upper := start + offset
		lower := upper + halfLength
		tempReal := currentReal*realOut[lower] - currentImag*imagOut[lower]
		tempImag := currentReal*imagOut[lower] + currentImag*realOut[lower]
		realOut[lower] = realOut[upper] - tempReal
		imagOut[lower] = imagOut[upper] - tempImag
		realOut[upper] += tempReal
		imagOut[upper] += tempImag
		nextReal := currentReal*twiddleStepReal - currentImag*twiddleStepImag
		nextImag := currentReal*twiddleStepImag + currentImag*twiddleStepReal
		currentReal, currentImag = nextReal, nextImag
	}
}

func physicsNaiveDFT(realOut []float32, imagOut []float32, inverse bool) {
	count := len(realOut)
	inputReal := append([]float32(nil), realOut...)
	inputImag := append([]float32(nil), imagOut...)
	sign := -1.0
	if inverse {
		sign = 1.0
	}

	for outputIndex := range count {
		realOut[outputIndex], imagOut[outputIndex] = physicsNaiveDFTValue(
			inputReal, inputImag, sign, outputIndex,
		)
	}
}

func physicsNaiveDFTValue(
	inputReal []float32,
	inputImag []float32,
	sign float64,
	outputIndex int,
) (float32, float32) {
	var sumReal float32
	var sumImag float32
	count := len(inputReal)

	for sourceIndex := range count {
		angle := sign * 2 * math.Pi * float64(outputIndex) * float64(sourceIndex) / float64(count)
		cosine := float32(math.Cos(angle))
		sine := float32(math.Sin(angle))
		sumReal += inputReal[sourceIndex]*cosine - inputImag[sourceIndex]*sine
		sumImag += inputReal[sourceIndex]*sine + inputImag[sourceIndex]*cosine
	}

	return sumReal, sumImag
}
