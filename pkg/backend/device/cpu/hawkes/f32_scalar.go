package hawkes

func HawkesKernelMatrixScalar(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	eventCount := len(eventTimes)

	for rowIndex := 0; rowIndex < eventCount; rowIndex++ {
		for colIndex := 0; colIndex < eventCount; colIndex++ {
			if colIndex >= rowIndex {
				out[rowIndex*eventCount+colIndex] = 0
				continue
			}

			delta := eventTimes[rowIndex] - eventTimes[colIndex]
			out[rowIndex*eventCount+colIndex] = alpha * hawkesExpScalar(-beta*delta)
		}
	}
}
