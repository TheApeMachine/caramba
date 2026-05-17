package dtype

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewBfloat16FromFloat32(test *testing.T) {
	Convey("Given a float32 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should create a BF16 value", func() {
			So(bf16, ShouldEqual, BF16(0x3f80))
			So((&bf16).Float32(), ShouldEqual, 1.0)
		})
	})
}

func TestNewBfloat16FromBytes(test *testing.T) {
	Convey("Given a byte slice", test, func() {
		bf16 := NewBfloat16FromBytes([]byte{0x3f, 0x80})

		Convey("It should create a BF16 value", func() {
			So(bf16, ShouldEqual, BF16(0x3f80))
			So((&bf16).Float32(), ShouldEqual, 1.0)
		})
	})
}

func TestBF16_Bytes(test *testing.T) {
	Convey("Given a BF16 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should return the bytes value", func() {
			So(bf16.Bytes(), ShouldResemble, []byte{0x3f, 0x80})
		})
	})
}

func TestBF16_Decode(test *testing.T) {
	Convey("Given a byte slice", test, func() {
		bf16 := NewBfloat16FromBytes([]byte{0x3f, 0x80})

		Convey("It should decode the byte slice into a BF16 value", func() {
			So(bf16.Decode([]byte{0x3f, 0x80}), ShouldResemble, []BF16{bf16})
		})
	})
}

func TestBF16_Encode(test *testing.T) {
	Convey("Given a BF16 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should encode the BF16 value into a byte slice", func() {
			So(bf16.Encode([]BF16{bf16}), ShouldResemble, []byte{0x3f, 0x80})
		})
	})
}

func TestBF16_DecodeFloat32(test *testing.T) {
	Convey("Given a byte slice", test, func() {
		bf16 := NewBfloat16FromBytes([]byte{0x3f, 0x80})

		Convey("It should decode the byte slice into a float32 value", func() {
			So(bf16.DecodeFloat32([]byte{0x3f, 0x80}), ShouldResemble, []float32{1.0})
		})
	})
}

func TestBF16_EncodeFloat32(test *testing.T) {
	Convey("Given a float32 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should encode the float32 value into a byte slice", func() {
			So(bf16.EncodeFloat32([]float32{1.0}), ShouldResemble, []byte{0x3f, 0x80})
		})
	})
}

func TestBF16_Float32(test *testing.T) {
	Convey("Given a BF16 value", test, func() {
		bf16 := NewBfloat16FromFloat32(1.0)

		Convey("It should return the float32 value", func() {
			So(bf16.Float32(), ShouldEqual, 1.0)
		})
	})
}

func BenchmarkBF16_NewBfloat16FromFloat32(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewBfloat16FromFloat32(1.0)
	}
}

func BenchmarkBF16_NewBfloat16FromBytes(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = NewBfloat16FromBytes([]byte{0x3f, 0x80})
	}
}

func BenchmarkBF16_Bytes(benchmark *testing.B) {
	bf16 := NewBfloat16FromFloat32(1.0)

	for benchmark.Loop() {
		_ = bf16.Bytes()
	}
}

func BenchmarkBF16_Decode(benchmark *testing.B) {
	bf16 := NewBfloat16FromBytes([]byte{0x3f, 0x80})

	for benchmark.Loop() {
		_ = bf16.Decode([]byte{0x3f, 0x80})
	}
}

func BenchmarkBF16_Encode(benchmark *testing.B) {
	bf16 := NewBfloat16FromFloat32(1.0)

	for benchmark.Loop() {
		_ = bf16.Encode([]BF16{bf16})
	}
}

func BenchmarkBF16_DecodeFloat32(benchmark *testing.B) {
	bf16 := NewBfloat16FromBytes([]byte{0x3f, 0x80})

	for benchmark.Loop() {
		_ = bf16.DecodeFloat32([]byte{0x3f, 0x80})
	}
}

func BenchmarkBF16_EncodeFloat32(benchmark *testing.B) {
	bf16 := NewBfloat16FromFloat32(1.0)

	for benchmark.Loop() {
		_ = bf16.EncodeFloat32([]float32{1.0})
	}
}

func BenchmarkBF16_Float32(benchmark *testing.B) {
	bf16 := NewBfloat16FromFloat32(1.0)

	for benchmark.Loop() {
		_ = bf16.Float32()
	}
}
