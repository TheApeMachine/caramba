//go:build arm64

package hawkes

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const hawkesNEONExpVectorMaxULP = 4

const hawkesNEONCompositeMaxULP = 4

func TestHawkesExpSumFloat32NEONParity(t *testing.T) {
	convey.Convey("Given HawkesExpSumNEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar exp sum for N=%d", length), func() {
				exponents := randomHawkesExponents(length, 0x2600+int64(length))
				want := hawkesExpSumReferenceNEON(exponents)
				got := HawkesExpSumNEON(exponents, length)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, hawkesNEONExpVectorMaxULP)
			})
		}
	})
}

func TestHawkesExpSumFloat32NEONAsmDirectParity(t *testing.T) {
	convey.Convey("Given HawkesExpSumNEONAsm block multiples of four", t, func() {
		for _, length := range parity.Lengths {
			if length&3 != 0 {
				continue
			}

			convey.Convey(fmt.Sprintf("It should match scalar exp sum for N=%d", length), func() {
				exponents := randomHawkesExponents(length, 0x2608+int64(length))
				want := hawkesExpSumReferenceNEON(exponents)
				got := HawkesExpSumNEONAsm(&exponents[0], length)

				parity.AssertFloat32SlicesWithinULP(t, []float32{got}, []float32{want}, hawkesNEONExpVectorMaxULP)
			})
		}
	})
}

func TestHawkesScaledExpStoreFloat32NEONParity(t *testing.T) {
	convey.Convey("Given HawkesScaledExpStoreNEON", t, func() {
		alpha := float32(0.5)

		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar scaled exp for N=%d", length), func() {
				exponents := randomHawkesExponents(length, 0x2610+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				for index, value := range exponents {
					want[index] = alpha * hawkesExpScalar(value)
				}

				HawkesScaledExpStoreNEON(exponents, alpha, got, length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesNEONExpVectorMaxULP)
			})
		}
	})
}

func TestHawkesScaledExpStoreFloat32NEONAsmDirectParity(t *testing.T) {
	convey.Convey("Given HawkesScaledExpStoreNEONAsm block multiples of four", t, func() {
		alpha := float32(0.5)

		for _, length := range parity.Lengths {
			if length&3 != 0 {
				continue
			}

			convey.Convey(fmt.Sprintf("It should match scalar scaled exp for N=%d", length), func() {
				exponents := randomHawkesExponents(length, 0x2618+int64(length))
				want := make([]float32, length)
				got := make([]float32, length)

				for index, value := range exponents {
					want[index] = alpha * hawkesExpScalar(value)
				}

				HawkesScaledExpStoreNEONAsm(&exponents[0], alpha, &got[0], length)

				parity.AssertFloat32SlicesWithinULP(t, got, want, hawkesNEONExpVectorMaxULP)
			})
		}
	})
}
