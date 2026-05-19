//go:build arm64

package hawkes

import (
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func HawkesIntensityNative(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	scratch := BorrowFloat32Buffer(len(eventTimes))
	defer ReleaseFloat32Buffer(scratch)

	for queryIndex, queryTime := range queryTimes {
		out[queryIndex] = mu + hawkesExcitationAtNEON(
			queryTime, eventTimes, scratch, alpha, beta,
		)
	}
}

func hawkesExcitationAtNEON(
	queryTime float32,
	eventTimes, scratch []float32,
	alpha, beta float32,
) float32 {
	validCount := 0

	for _, eventTime := range eventTimes {
		if eventTime > queryTime {
			continue
		}

		scratch[validCount] = -beta * (queryTime - eventTime)
		validCount++
	}

	if validCount == 0 {
		return 0
	}

	blockCount := validCount &^ 3
	sum := float32(0)

	if blockCount > 0 {
		sum = HawkesExpSumNEONAsm(&scratch[0], blockCount)
	}

	for index := blockCount; index < validCount; index++ {
		sum += hawkesExpScalar(scratch[index])
	}

	return alpha * sum
}

func HawkesKernelMatrixNative(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	eventCount := len(eventTimes)
	scratch := BorrowFloat32Buffer(eventCount)
	defer ReleaseFloat32Buffer(scratch)

	for rowIndex := 0; rowIndex < eventCount; rowIndex++ {
		rowStart := rowIndex * eventCount

		for colIndex := rowIndex; colIndex < eventCount; colIndex++ {
			out[rowStart+colIndex] = 0
		}

		if rowIndex == 0 {
			continue
		}

		for colIndex := 0; colIndex < rowIndex; colIndex++ {
			scratch[colIndex] = -beta * (eventTimes[rowIndex] - eventTimes[colIndex])
		}

		blockPrefix := rowIndex &^ 3

		if blockPrefix > 0 {
			HawkesScaledExpStoreNEONAsm(
				&scratch[0], alpha, &out[rowStart], blockPrefix,
			)
		}

		for colIndex := blockPrefix; colIndex < rowIndex; colIndex++ {
			out[rowStart+colIndex] = alpha * hawkesExpScalar(scratch[colIndex])
		}
	}
}

func hawkesExpScalar(value float32) float32 {
	scratch := [1]float32{value}
	activation.Exp(
		unsafe.Pointer(unsafe.SliceData(scratch[:])),
		unsafe.Pointer(unsafe.SliceData(scratch[:])),
		len(scratch),
		dtype.Float32,
	)

	return scratch[0]
}

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
	var mutualInformation float64

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			jointValue := float64(joint[xIndex*yCount+yIndex])

			if jointValue <= epsilon {
				continue
			}

			denominator := marginalX[xIndex]*marginalY[yIndex] + epsilon
			mutualInformation += jointValue * math.Log(jointValue/denominator)
		}
	}

	out[0] = float32(mutualInformation)
}
