//go:build arm64

package cpu

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
	ExpFloat32Native(scratch[:], scratch[:])

	return scratch[0]
}
