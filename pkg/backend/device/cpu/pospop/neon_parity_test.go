//go:build arm64

package pospop

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestCount8NEONParity(t *testing.T) {
	convey.Convey("Given Count8NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count8Generic for N=%d", length), func() {
				buffer := make([]uint8, length)
				fillUint8Buffer(buffer, length)

				var want, got [8]int
				Count8Generic(&want, buffer)
				Count8NEON(&got, buffer)

				assertCount8Equal(t, &want, &got)
			})
		}
	})
}

func TestCount16NEONParity(t *testing.T) {
	convey.Convey("Given Count16NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count16Generic for N=%d", length), func() {
				buffer := make([]uint16, length)
				fillUint16Buffer(buffer, length)

				var want, got [16]int
				Count16Generic(&want, buffer)
				Count16NEON(&got, buffer)

				assertCount16Equal(t, &want, &got)
			})
		}
	})
}

func TestCount32NEONParity(t *testing.T) {
	convey.Convey("Given Count32NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count32Generic for N=%d", length), func() {
				buffer := make([]uint32, length)
				fillUint32Buffer(buffer, length)

				var want, got [32]int
				Count32Generic(&want, buffer)
				Count32NEON(&got, buffer)

				assertCount32Equal(t, &want, &got)
			})
		}
	})
}

func TestCount64NEONParity(t *testing.T) {
	convey.Convey("Given Count64NEON", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match Count64Generic for N=%d", length), func() {
				buffer := make([]uint64, length)
				fillUint64Buffer(buffer, length)

				var want, got [64]int
				Count64Generic(&want, buffer)
				Count64NEON(&got, buffer)

				assertCount64Equal(t, &want, &got)
			})
		}
	})
}

func BenchmarkCount8NEON(b *testing.B) {
	buffer := make([]uint8, 8192)
	fillUint8Buffer(buffer, 0)

	var counts [8]int

	b.ResetTimer()
	for b.Loop() {
		Count8NEON(&counts, buffer)
	}
}
