//go:build ignore

package main

import (
	"fmt"
	"math"
	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

const (
	mantBits64 = 52
	expBits64  = 11
	bias64     = 1023
	nan64      = (1<<expBits64 - 1) << mantBits64
	inf64      = (1<<expBits64 - 1) << mantBits64
	neg64      = 1 << (expBits64 + mantBits64)
)

func unpack64(value uint64) (sign, mant uint64, exp int, inf, nan bool) {
	sign = value & neg64
	mant = value & (1<<mantBits64 - 1)
	exp = int(value>>mantBits64) & (1<<expBits64 - 1)
	switch exp {
	case 1<<expBits64 - 1:
		if mant != 0 {
			return sign, mant, exp, false, true
		}
		return sign, mant, exp, true, false
	case 0:
		if mant != 0 {
			exp = bias64 + 1
			for mant < 1<<mantBits64 {
				mant <<= 1
				exp--
			}
		}
	default:
		mant |= 1 << mantBits64
		exp += bias64
	}
	return sign, mant, exp, inf, nan
}

func pack64(sign, mant uint64, exp int, trunc uint64) uint64 {
	if mant == 0 {
		return sign
	}
	for mant < 1<<mantBits64 {
		mant <<= 1
		exp--
	}
	for mant >= 4<<mantBits64 {
		trunc |= mant & 1
		mant >>= 1
		exp++
	}
	if mant >= 2<<mantBits64 {
		if mant&1 != 0 && (trunc != 0 || mant&2 != 0) {
			mant++
			if mant >= 4<<mantBits64 {
				mant >>= 1
				exp++
			}
		}
		mant >>= 1
		exp++
	}
	if exp >= 1<<expBits64-1+bias64 {
		return sign ^ inf64
	}
	if exp < bias64+1 {
		if exp < bias64-int(mantBits64) {
			return sign
		}
	}
	return sign | uint64(exp-bias64)<<mantBits64 | mant&(1<<mantBits64-1)
}

func f32to64(value float32) uint64 {
	const delta = mantBits64 - 23
	bits := math.Float32bits(value)
	sign := uint64(bits&0x80000000) << 32
	exp := int(bits>>23) & 0xFF
	mant := uint64(bits & 0x7FFFFF)
	if exp == 255 {
		if mant != 0 {
			return sign | nan64
		}
		return sign | inf64
	}
	if exp == 0 {
		if mant == 0 {
			return sign
		}
		mant <<= delta
		exp64 := bias64 + 1
		for mant < 1<<mantBits64 {
			mant <<= 1
			exp64--
		}
		return pack64(sign, mant, exp64, 0)
	}
	return pack64(sign, mant<<delta, exp-127+bias64+1, 0)
}

func geluF64(value, erfValue float32) float32 {
	x64 := math.Float64frombits(f32to64(value))
	e64 := math.Float64frombits(f32to64(erfValue))
	return float32(0.5 * x64 * (1 + e64))
}

func main() {
	x := float32(-3.9166667)
	t := x * 0.7071067811865475
	erf := float32(math.Erf(float64(t)))
	got := geluF64(x, erf)
	ref := cpumath.FastGelu32(x)
	fmt.Printf("ref=%g got=%g ulp=%d\n", ref, got, parity.Float32ULPDistance(ref, got))
}
