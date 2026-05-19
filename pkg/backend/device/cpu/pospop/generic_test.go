package pospop

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestCount8Generic(t *testing.T) {
	convey.Convey("Given Count8Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should count set bits for N=%d", length), func() {
				buffer := make([]uint8, length)
				fillUint8Buffer(buffer, length)

				var counts [8]int
				Count8Generic(&counts, buffer)

				var expect [8]int
				for _, value := range buffer {
					for bit := range 8 {
						expect[bit] += int(value >> bit & 1)
					}
				}

				assertCount8Equal(t, &expect, &counts)
			})
		}
	})
}

func TestCount16Generic(t *testing.T) {
	convey.Convey("Given Count16Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should count set bits for N=%d", length), func() {
				buffer := make([]uint16, length)
				fillUint16Buffer(buffer, length)

				var counts [16]int
				Count16Generic(&counts, buffer)

				var expect [16]int
				for _, value := range buffer {
					for bit := range 16 {
						expect[bit] += int(value >> bit & 1)
					}
				}

				assertCount16Equal(t, &expect, &counts)
			})
		}
	})
}

func TestCount32Generic(t *testing.T) {
	convey.Convey("Given Count32Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should count set bits for N=%d", length), func() {
				buffer := make([]uint32, length)
				fillUint32Buffer(buffer, length)

				var counts [32]int
				Count32Generic(&counts, buffer)

				var expect [32]int
				for _, value := range buffer {
					for bit := range 32 {
						expect[bit] += int(value >> bit & 1)
					}
				}

				assertCount32Equal(t, &expect, &counts)
			})
		}
	})
}

func TestCount64Generic(t *testing.T) {
	convey.Convey("Given Count64Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should count set bits for N=%d", length), func() {
				buffer := make([]uint64, length)
				fillUint64Buffer(buffer, length)

				var counts [64]int
				Count64Generic(&counts, buffer)

				var expect [64]int
				for _, value := range buffer {
					for bit := range 64 {
						expect[bit] += int(value >> bit & 1)
					}
				}

				assertCount64Equal(t, &expect, &counts)
			})
		}
	})
}

func BenchmarkCount8Generic(b *testing.B) {
	buffer := make([]uint8, 8192)
	fillUint8Buffer(buffer, 0)

	var counts [8]int

	b.ResetTimer()
	for b.Loop() {
		Count8Generic(&counts, buffer)
	}
}

func BenchmarkCount16Generic(b *testing.B) {
	buffer := make([]uint16, 8192)
	fillUint16Buffer(buffer, 0)

	var counts [16]int

	b.ResetTimer()
	for b.Loop() {
		Count16Generic(&counts, buffer)
	}
}

func BenchmarkCount32Generic(b *testing.B) {
	buffer := make([]uint32, 8192)
	fillUint32Buffer(buffer, 0)

	var counts [32]int

	b.ResetTimer()
	for b.Loop() {
		Count32Generic(&counts, buffer)
	}
}

func BenchmarkCount64Generic(b *testing.B) {
	buffer := make([]uint64, 8192)
	fillUint64Buffer(buffer, 0)

	var counts [64]int

	b.ResetTimer()
	for b.Loop() {
		Count64Generic(&counts, buffer)
	}
}
