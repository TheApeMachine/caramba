package hawkes

import "github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"

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
	return parity.Lengths
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

/*
hawkesExpSumReferenceNEON matches HawkesExpSumNEONAsm reduction order:
four lane-wise partial float32 sums, then a left-to-right combine.
*/
func hawkesExpSumReferenceNEON(exponents []float32) float32 {
	var lanePartial [4]float32

	for index, value := range exponents {
		laneIndex := index & 3
		lanePartial[laneIndex] += hawkesExpScalar(value)
	}

	return lanePartial[0] + lanePartial[1] + lanePartial[2] + lanePartial[3]
}

/*
hawkesExpSumReferenceAVX512 matches HawkesExpSumFloat32AVX512Asm ymm accumulation
and final horizontal reduction order.
*/
func hawkesExpSumReferenceAVX512(exponents []float32) float32 {
	const laneCount = 16
	var lanePartial [laneCount]float32

	for index, value := range exponents {
		laneIndex := index & (laneCount - 1)
		lanePartial[laneIndex] += hawkesExpScalar(value)
	}

	sum := lanePartial[0]
	sum += lanePartial[1]
	sum += lanePartial[2]
	sum += lanePartial[3]
	sum += lanePartial[4]
	sum += lanePartial[5]
	sum += lanePartial[6]
	sum += lanePartial[7]
	sum += lanePartial[8]
	sum += lanePartial[9]
	sum += lanePartial[10]
	sum += lanePartial[11]
	sum += lanePartial[12]
	sum += lanePartial[13]
	sum += lanePartial[14]
	sum += lanePartial[15]

	return sum
}
