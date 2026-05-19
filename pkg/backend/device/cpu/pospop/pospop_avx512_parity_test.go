//go:build amd64

package pospop

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512PospopAvailable() bool {
	return cpu.X86.HasBMI2 && cpu.X86.HasAVX512BW
}

func TestCount8AVX512Parity(t *testing.T) {
	if !avx512PospopAvailable() {
		t.Skip("AVX-512 BW and BMI2 required")
	}

	convey.Convey("Given Count8AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count8Generic for N=%d", length), func() {
				buffer := make([]uint8, length)
				fillUint8Buffer(buffer, length)

				var want, got [8]int
				Count8Generic(&want, buffer)
				Count8AVX512(&got, buffer)

				assertCount8Equal(t, &want, &got)
			})
		}
	})
}

func TestCount16AVX512Parity(t *testing.T) {
	if !avx512PospopAvailable() {
		t.Skip("AVX-512 BW and BMI2 required")
	}

	convey.Convey("Given Count16AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count16Generic for N=%d", length), func() {
				buffer := make([]uint16, length)
				fillUint16Buffer(buffer, length)

				var want, got [16]int
				Count16Generic(&want, buffer)
				Count16AVX512(&got, buffer)

				assertCount16Equal(t, &want, &got)
			})
		}
	})
}

func TestCount32AVX512Parity(t *testing.T) {
	if !avx512PospopAvailable() {
		t.Skip("AVX-512 BW and BMI2 required")
	}

	convey.Convey("Given Count32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count32Generic for N=%d", length), func() {
				buffer := make([]uint32, length)
				fillUint32Buffer(buffer, length)

				var want, got [32]int
				Count32Generic(&want, buffer)
				Count32AVX512(&got, buffer)

				assertCount32Equal(t, &want, &got)
			})
		}
	})
}

func TestCount64AVX512Parity(t *testing.T) {
	if !avx512PospopAvailable() {
		t.Skip("AVX-512 BW and BMI2 required")
	}

	convey.Convey("Given Count64AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count64Generic for N=%d", length), func() {
				buffer := make([]uint64, length)
				fillUint64Buffer(buffer, length)

				var want, got [64]int
				Count64Generic(&want, buffer)
				Count64AVX512(&got, buffer)

				assertCount64Equal(t, &want, &got)
			})
		}
	})
}
