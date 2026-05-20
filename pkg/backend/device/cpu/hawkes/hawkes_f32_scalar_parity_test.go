package hawkes

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const hawkesScalarMaxULP = 4

func TestHawkesIntensityScalarParityLengths(t *testing.T) {
	convey.Convey("Given HawkesIntensityScalar", t, func() {
		for _, eventCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2300+int64(eventCount))
				queryTimes := hawkesQueryTimesForTest(eventCount, 0x2301+int64(eventCount))
				first := make([]float32, eventCount)
				second := make([]float32, eventCount)

				HawkesIntensityScalar(eventTimes, queryTimes, first, 0.1, 0.5, 1.0)
				HawkesIntensityScalar(eventTimes, queryTimes, second, 0.1, 0.5, 1.0)

				parity.AssertFloat32SlicesWithinULP(t, first, second, 0)
			})
		}
	})
}

func TestHawkesLogLikelihoodScalarParityLengths(t *testing.T) {
	convey.Convey("Given HawkesLogLikelihoodScalar", t, func() {
		for _, eventCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match itself for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2310+int64(eventCount))
				first := make([]float32, 1)
				second := make([]float32, 1)

				HawkesLogLikelihoodScalar(eventTimes, 10.0, 0.2, 0.5, 1.0, first)
				HawkesLogLikelihoodScalar(eventTimes, 10.0, 0.2, 0.5, 1.0, second)

				parity.AssertFloat32SlicesWithinULP(t, first, second, 0)
			})
		}
	})
}

func TestHawkesKernelMatrixScalarParityLengths(t *testing.T) {
	convey.Convey("Given hawkesKernelMatrixScalar", t, func() {
		for _, eventCount := range hawkesKernelMatrixParityEventCounts() {
			convey.Convey(fmt.Sprintf("It should match itself for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2320+int64(eventCount))
				first := make([]float32, eventCount*eventCount)
				second := make([]float32, eventCount*eventCount)

				hawkesKernelMatrixScalar(eventTimes, first, 0.5, 1.0)
				hawkesKernelMatrixScalar(eventTimes, second, 0.5, 1.0)

				parity.AssertFloat32SlicesWithinULP(t, first, second, 0)
			})
		}
	})
}

func TestMarkovMutualInformationScalarParityLengths(t *testing.T) {
	convey.Convey("Given MarkovMutualInformationScalar", t, func() {
		for _, length := range parity.Lengths {
			side := hawkesMarkovSide(length)
			convey.Convey(fmt.Sprintf("It should match itself for N=%d side=%d", length, side), func() {
				joint := randomHawkesJoint(side, side, 0x2330+int64(length))
				first := make([]float32, 1)
				second := make([]float32, 1)

				MarkovMutualInformationScalar(joint, side, side, first)
				MarkovMutualInformationScalar(joint, side, side, second)

				parity.AssertFloat32SlicesWithinULP(t, first, second, 0)
			})
		}
	})
}

func TestHawkesKernelMatrixNativeVsScalarParityLengths(t *testing.T) {
	convey.Convey("Given HawkesKernelMatrixNative vs hawkesKernelMatrixScalar", t, func() {
		for _, eventCount := range hawkesKernelMatrixParityEventCounts() {
			convey.Convey(fmt.Sprintf("It should match scalar for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2338+int64(eventCount))
				got := make([]float32, eventCount*eventCount)
				want := make([]float32, eventCount*eventCount)

				HawkesKernelMatrixNative(eventTimes, got, 0.5, 1.0)
				hawkesKernelMatrixScalar(eventTimes, want, 0.5, 1.0)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesScalarMaxULP)
			})
		}
	})
}

func TestHawkesIntensityNativeVsScalarParityLengths(t *testing.T) {
	convey.Convey("Given HawkesIntensityNative vs HawkesIntensityScalar", t, func() {
		for _, eventCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2340+int64(eventCount))
				queryTimes := hawkesSingleQueryAfterEvents(eventTimes, 0x2341+int64(eventCount))
				got := make([]float32, 1)
				want := make([]float32, 1)

				HawkesIntensityNative(eventTimes, queryTimes, got, 0.1, 0.5, 1.0)
				HawkesIntensityScalar(eventTimes, queryTimes, want, 0.1, 0.5, 1.0)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesScalarMaxULP)
			})
		}
	})
}

func TestHawkesLogLikelihoodNativeVsScalarParityLengths(t *testing.T) {
	convey.Convey("Given HawkesLogLikelihoodNative vs HawkesLogLikelihoodScalar", t, func() {
		for _, eventCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2350+int64(eventCount))
				got := make([]float32, 1)
				want := make([]float32, 1)

				HawkesLogLikelihoodNative(eventTimes, 10.0, 0.2, 0.5, 1.0, got)
				HawkesLogLikelihoodScalar(eventTimes, 10.0, 0.2, 0.5, 1.0, want)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesScalarMaxULP)
			})
		}
	})
}

func hawkesMarkovSide(length int) int {
	if length < 4 {
		return length
	}

	if length > 64 {
		return 8
	}

	return 4
}

func randomHawkesJoint(xCount, yCount int, seed int64) []float32 {
	joint := make([]float32, xCount*yCount)
	state := uint64(seed)
	total := float32(0)

	for index := range joint {
		state = state*6364136223846793005 + 1442695040888963407
		value := float32(state>>40) * 1e-3
		joint[index] = value
		total += value
	}

	for index := range joint {
		joint[index] /= total
	}

	return joint
}
