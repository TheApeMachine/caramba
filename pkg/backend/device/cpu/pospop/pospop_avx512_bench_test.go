//go:build amd64

package pospop

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkCount8AVX512(b *testing.B) {
	if !cpu.X86.HasBMI2 || !cpu.X86.HasAVX512BW {
		b.Skip("AVX-512 BW and BMI2 required")
	}

	buffer := make([]uint8, 8192)
	fillUint8Buffer(buffer, 0)

	var counts [8]int

	b.ResetTimer()
	for b.Loop() {
		Count8AVX512(&counts, buffer)
	}
}

func BenchmarkCount16AVX512(b *testing.B) {
	if !cpu.X86.HasBMI2 || !cpu.X86.HasAVX512BW {
		b.Skip("AVX-512 BW and BMI2 required")
	}

	buffer := make([]uint16, 8192)
	fillUint16Buffer(buffer, 0)

	var counts [16]int

	b.ResetTimer()
	for b.Loop() {
		Count16AVX512(&counts, buffer)
	}
}

func BenchmarkCount32AVX512(b *testing.B) {
	if !cpu.X86.HasBMI2 || !cpu.X86.HasAVX512BW {
		b.Skip("AVX-512 BW and BMI2 required")
	}

	buffer := make([]uint32, 8192)
	fillUint32Buffer(buffer, 0)

	var counts [32]int

	b.ResetTimer()
	for b.Loop() {
		Count32AVX512(&counts, buffer)
	}
}

func BenchmarkCount64AVX512(b *testing.B) {
	if !cpu.X86.HasBMI2 || !cpu.X86.HasAVX512BW {
		b.Skip("AVX-512 BW and BMI2 required")
	}

	buffer := make([]uint64, 8192)
	fillUint64Buffer(buffer, 0)

	var counts [64]int

	b.ResetTimer()
	for b.Loop() {
		Count64AVX512(&counts, buffer)
	}
}
