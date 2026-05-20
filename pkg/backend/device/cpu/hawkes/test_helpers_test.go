package hawkes

func randomHawkesExponents(length int, seed int64) []float32 {
	values := make([]float32, length)
	state := uint64(seed)

	for index := range values {
		state = state*6364136223846793005 + 1442695040888963407
		mantissa := float64(state>>11) / float64(1<<53)
		values[index] = float32(-0.5 + mantissa)
	}

	return values
}

func hawkesJitterFloat32(state uint64) float32 {
	return float32((state >> 40) & 0xFFFFF) * 1e-6
}

func hawkesEventTimesForTest(eventCount int, seed int64) []float32 {
	eventTimes := make([]float32, eventCount)
	state := uint64(seed)
	eventTimes[0] = hawkesJitterFloat32(state)

	for index := 1; index < eventCount; index++ {
		state = state*6364136223846793005 + 1442695040888963407
		eventTimes[index] = eventTimes[index-1] + 0.25 + hawkesJitterFloat32(state)
	}

	return eventTimes
}

func hawkesSingleQueryAfterEvents(eventTimes []float32, seed int64) []float32 {
	queryTimes := make([]float32, 1)
	lastEvent := eventTimes[len(eventTimes)-1]
	state := uint64(seed)

	state = state*6364136223846793005 + 1442695040888963407
	queryTimes[0] = lastEvent + 0.5 + hawkesJitterFloat32(state)

	return queryTimes
}

func hawkesKernelMatrixParityEventCounts() []int {
	return []int{1, 7, 64, 128}
}

func hawkesQueryTimesForTest(queryCount int, seed int64) []float32 {
	queryTimes := make([]float32, queryCount)
	state := uint64(seed + 17)

	for index := range queryTimes {
		state = state*6364136223846793005 + 1442695040888963407
		queryTimes[index] = float32(index)*0.25 + 0.1 + hawkesJitterFloat32(state)
	}

	return queryTimes
}

func hawkesExpSumReference(exponents []float32) float32 {
	sum := float32(0)

	for _, value := range exponents {
		sum += hawkesExpScalar(value)
	}

	return sum
}
