//go:build arm64

package cpu

import (
	"math"
)

func HawkesLogLikelihoodNative(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	eventCount := len(eventTimes)
	scratch := BorrowFloat32Buffer(eventCount)
	defer ReleaseFloat32Buffer(scratch)

	var sumLog float64

	for eventIndex := range eventTimes {
		validCount := 0

		for previousIndex := 0; previousIndex < eventIndex; previousIndex++ {
			delta := eventTimes[eventIndex] - eventTimes[previousIndex]
			scratch[validCount] = -beta * delta
			validCount++
		}

		intensity := mu

		if validCount > 0 {
			blockCount := validCount &^ 3
			sum := float32(0)

			if blockCount > 0 {
				sum = HawkesExpSumNEONAsm(&scratch[0], blockCount)
			}

			for index := blockCount; index < validCount; index++ {
				sum += hawkesExpScalar(scratch[index])
			}

			intensity += alpha * sum
		}

		sumLog += math.Log(math.Max(1e-12, float64(intensity)))
	}

	compensator := float64(mu * totalT)

	for _, eventTime := range eventTimes {
		compensator += float64(alpha/beta) * (1 - math.Exp(float64(-beta*(totalT-eventTime))))
	}

	out[0] = float32(sumLog - compensator)
}

func MarkovMutualInformationNative(joint []float32, xCount, yCount int, out []float32) {
	marginalX := make([]float64, xCount)
	marginalY := make([]float64, yCount)

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			value := float64(joint[xIndex*yCount+yIndex])
			marginalX[xIndex] += value
			marginalY[yIndex] += value
		}
	}

	const epsilon = 1e-12
	ratioScratch := BorrowFloat32Buffer(yCount)
	weightScratch := BorrowFloat32Buffer(yCount)
	defer ReleaseFloat32Buffer(ratioScratch)
	defer ReleaseFloat32Buffer(weightScratch)

	var mutualInformation float64

	for xIndex := 0; xIndex < xCount; xIndex++ {
		rowStart := xIndex * yCount
		rowView := joint[rowStart : rowStart+yCount]
		marginalXValue := float32(marginalX[xIndex])

		for yIndex := 0; yIndex < yCount; yIndex++ {
			jointValue := rowView[yIndex]
			ratioScratch[yIndex] = 1
			weightScratch[yIndex] = 0

			if jointValue <= epsilon {
				continue
			}

			denominator := marginalXValue*float32(marginalY[yIndex]) + epsilon
			ratioScratch[yIndex] = jointValue / denominator
			weightScratch[yIndex] = jointValue
		}

		blockCount := yCount &^ 3

		if blockCount > 0 {
			LogFloat32Native(ratioScratch[:blockCount], ratioScratch[:blockCount])
			MulFloat32Native(weightScratch[:blockCount], rowView[:blockCount], ratioScratch[:blockCount])
			mutualInformation += float64(SumFloat32Native(weightScratch[:blockCount]))
		}

		for yIndex := blockCount; yIndex < yCount; yIndex++ {
			if rowView[yIndex] <= epsilon {
				continue
			}

			mutualInformation += float64(rowView[yIndex]) *
				math.Log(float64(ratioScratch[yIndex]))
		}
	}

	out[0] = float32(mutualInformation)
}
