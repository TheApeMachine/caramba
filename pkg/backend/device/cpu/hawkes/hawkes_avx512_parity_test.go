//go:build amd64

package hawkes

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const hawkesAVX512ExpVectorMaxULP = 4

const hawkesAVX512CompositeMaxULP = 4

func avx512HawkesAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestHawkesExpSumFloat32AVX512Parity(t *testing.T) {
	if !avx512HawkesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given HawkesExpSumFloat32AVX512Asm", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar exp sum for N=%d", length), func() {
				exponents := randomHawkesExponents(length, 0x2400+int64(length))
				want := hawkesExpSumReferenceAVX512(exponents)
				got := HawkesExpSumFloat32AVX512Asm(&exponents[0], length)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, hawkesAVX512ExpVectorMaxULP)
			})
		}
	})
}

func TestHawkesScaledExpStoreFloat32AVX512Parity(t *testing.T) {
	if !avx512HawkesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given HawkesScaledExpStoreFloat32AVX512Asm", t, func() {
		alpha := float32(0.5)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar scaled exp for N=%d", length), func() {
				exponents := randomHawkesExponents(length, 0x2410+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				for index, value := range exponents {
					want[index] = alpha * hawkesExpScalar(value)
				}

				HawkesScaledExpStoreFloat32AVX512Asm(&exponents[0], alpha, &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesAVX512ExpVectorMaxULP)
			})
		}
	})
}

func TestHawkesIntensityNativeAVX512Parity(t *testing.T) {
	if !avx512HawkesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given HawkesIntensityNative on amd64", t, func() {
		for _, eventCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match HawkesIntensityScalar for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2420+int64(eventCount))
				queryTimes := hawkesSingleQueryAfterEvents(eventTimes, 0x2421+int64(eventCount))
				got := make([]float32, 1)
				want := make([]float32, 1)

				HawkesIntensityNative(eventTimes, queryTimes, got, 0.1, 0.5, 1.0)
				HawkesIntensityScalar(eventTimes, queryTimes, want, 0.1, 0.5, 1.0)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesAVX512CompositeMaxULP)
			})
		}
	})
}

func TestHawkesKernelMatrixNativeAVX512Parity(t *testing.T) {
	if !avx512HawkesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given HawkesKernelMatrixNative on amd64", t, func() {
		for _, eventCount := range hawkesKernelMatrixParityEventCounts() {
			convey.Convey(fmt.Sprintf("It should match hawkesKernelMatrixScalar for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2430+int64(eventCount))
				got := make([]float32, eventCount*eventCount)
				want := make([]float32, eventCount*eventCount)

				HawkesKernelMatrixNative(eventTimes, got, 0.5, 1.0)
				hawkesKernelMatrixScalar(eventTimes, want, 0.5, 1.0)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesAVX512CompositeMaxULP)
			})
		}
	})
}

func TestHawkesLogLikelihoodNativeAVX512Parity(t *testing.T) {
	if !avx512HawkesAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given HawkesLogLikelihoodNative on amd64", t, func() {
		for _, eventCount := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match HawkesLogLikelihoodScalar for eventCount=%d", eventCount), func() {
				eventTimes := hawkesEventTimesForTest(eventCount, 0x2440+int64(eventCount))
				got := make([]float32, 1)
				want := make([]float32, 1)

				HawkesLogLikelihoodNative(eventTimes, 10.0, 0.2, 0.5, 1.0, got)
				HawkesLogLikelihoodScalar(eventTimes, 10.0, 0.2, 0.5, 1.0, want)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesAVX512CompositeMaxULP)
			})
		}
	})
}
