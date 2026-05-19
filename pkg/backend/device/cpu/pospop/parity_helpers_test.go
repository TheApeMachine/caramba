package pospop

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func fillUint8Buffer(buffer []uint8, seed int) {
	for index := range buffer {
		buffer[index] = uint8(index*17 + seed*31)
	}
}

func fillUint16Buffer(buffer []uint16, seed int) {
	for index := range buffer {
		buffer[index] = uint16(index*17 + seed*31)
	}
}

func fillUint32Buffer(buffer []uint32, seed int) {
	for index := range buffer {
		buffer[index] = uint32(index*17 + seed*31)
	}
}

func fillUint64Buffer(buffer []uint64, seed int) {
	for index := range buffer {
		buffer[index] = uint64(index*17 + seed*31)
	}
}

func assertCount8Equal(testingTB *testing.T, want, got *[8]int) {
	testingTB.Helper()

	for index := range want {
		if want[index] != got[index] {
			convey.So(got[index], convey.ShouldEqual, want[index])
			return
		}
	}
}

func assertCount16Equal(testingTB *testing.T, want, got *[16]int) {
	testingTB.Helper()

	for index := range want {
		if want[index] != got[index] {
			convey.So(got[index], convey.ShouldEqual, want[index])
			return
		}
	}
}

func assertCount32Equal(testingTB *testing.T, want, got *[32]int) {
	testingTB.Helper()

	for index := range want {
		if want[index] != got[index] {
			convey.So(got[index], convey.ShouldEqual, want[index])
			return
		}
	}
}

func assertCount64Equal(testingTB *testing.T, want, got *[64]int) {
	testingTB.Helper()

	for index := range want {
		if want[index] != got[index] {
			convey.So(got[index], convey.ShouldEqual, want[index])
			return
		}
	}
}
