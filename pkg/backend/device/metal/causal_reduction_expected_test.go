package metal

import "math"

func ivExpected(instrument []float32, treatment []float32, outcome []float32) float32 {
	sumZ, sumX, sumY, sumZY, sumZX := ivReductionTotals(instrument, treatment, outcome)
	count := float32(len(instrument))
	denominator := sumZX - (sumZ*sumX)/count
	numerator := sumZY - (sumZ*sumY)/count

	if float32(math.Abs(float64(denominator))) < 1.0e-12 {
		return 0
	}

	return numerator / denominator
}

func ivReductionTotals(
	instrument []float32,
	treatment []float32,
	outcome []float32,
) (float32, float32, float32, float32, float32) {
	partialCount := causalExpectedPartialCount(len(instrument))
	scratch := make([][5]float32, partialCount)

	for groupIndex := range partialCount {
		scratch[groupIndex] = ivPartialForGroup(instrument, treatment, outcome, groupIndex)
	}

	return ivFinalizePartials(scratch)
}

func ivPartialForGroup(
	instrument []float32,
	treatment []float32,
	outcome []float32,
	groupIndex int,
) [5]float32 {
	values := [5][256]float32{}

	for threadIndex := range 256 {
		valueIndex := groupIndex*256 + threadIndex
		if valueIndex >= len(instrument) {
			continue
		}

		instrumentValue := instrument[valueIndex]
		treatmentValue := treatment[valueIndex]
		outcomeValue := outcome[valueIndex]
		values[0][threadIndex] = instrumentValue
		values[1][threadIndex] = treatmentValue
		values[2][threadIndex] = outcomeValue
		values[3][threadIndex] = instrumentValue * outcomeValue
		values[4][threadIndex] = instrumentValue * treatmentValue
	}

	return reduceFiveArrays(values)
}

func ivFinalizePartials(scratch [][5]float32) (float32, float32, float32, float32, float32) {
	values := [5][256]float32{}

	for threadIndex := range 256 {
		for partialIndex := threadIndex; partialIndex < len(scratch); partialIndex += 256 {
			for valueIndex := range 5 {
				values[valueIndex][threadIndex] += scratch[partialIndex][valueIndex]
			}
		}
	}

	reduced := reduceFiveArrays(values)
	return reduced[0], reduced[1], reduced[2], reduced[3], reduced[4]
}

func reduceFiveArrays(values [5][256]float32) [5]float32 {
	for stride := 128; stride > 0; stride >>= 1 {
		for threadIndex := 0; threadIndex < stride; threadIndex++ {
			for valueIndex := range 5 {
				values[valueIndex][threadIndex] += values[valueIndex][threadIndex+stride]
			}
		}
	}

	return [5]float32{values[0][0], values[1][0], values[2][0], values[3][0], values[4][0]}
}

func dagExpected(conditionals []float32) float32 {
	partialCount := causalExpectedPartialCount(len(conditionals))
	partials := make([]float32, partialCount)

	for groupIndex := range partialCount {
		partials[groupIndex] = dagPartialForGroup(conditionals, groupIndex)
	}

	return dagFinalizePartials(partials)
}

func dagPartialForGroup(conditionals []float32, groupIndex int) float32 {
	values := [256]float32{}

	for threadIndex := range 256 {
		valueIndex := groupIndex*256 + threadIndex
		if valueIndex < len(conditionals) {
			values[threadIndex] = max(conditionals[valueIndex], 1.0e-12)
			continue
		}

		values[threadIndex] = 1
	}

	return reduceProductArray(values)
}

func dagFinalizePartials(partials []float32) float32 {
	values := [256]float32{}

	for threadIndex := range 256 {
		values[threadIndex] = 1
		for partialIndex := threadIndex; partialIndex < len(partials); partialIndex += 256 {
			values[threadIndex] *= partials[partialIndex]
		}
	}

	return reduceProductArray(values)
}

func reduceProductArray(values [256]float32) float32 {
	for stride := 128; stride > 0; stride >>= 1 {
		for threadIndex := 0; threadIndex < stride; threadIndex++ {
			values[threadIndex] *= values[threadIndex+stride]
		}
	}

	return values[0]
}

func causalExpectedPartialCount(count int) int {
	return (count + 255) / 256
}
