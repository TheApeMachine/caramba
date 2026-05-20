//go:build ignore

package main

import (
	"fmt"
	"math"
)

const (
	ln2Hi = 0x3FE62E42FEE00000
	ln2Lo = 0x3DEA39EF35793C76
	log2e = 0x3FF71547652B82FE
	overflow  = 0x40862E42FEFA39EF
	underflow = 0xC0874910D52D5AF7
	half64    = 0x3FE0000000000000
	one       = 0x3FF0000000000000
	zero      = 0
	expP1     = 0x3FC5555555555555
	expP2     = 0xBF66C16C16BEBD93
	expP3     = 0x3F11566AAF25DE2C
	expP4     = 0xBEBBBD41C5D26BF1
	expP5     = 0x3E66376972BEA4D0
	two64      = 0x4000000000000000
	neg05625   = 0xBFE2000000000000
	ra0        = 0xBF843412600D6435
	sa1        = 0x4033A6B9BD707687
)

func ldexp64(value uint64, exponent int) uint64 {
	fs, fm, fe, fi, fn := funpack64(value)
	if fi || fn {
		return value
	}
	return fpack64(fs, fm, fe+exponent, 0)
}

func sf64Neg(value uint64) uint64 { return value ^ (1 << 63) }

func sf64Lt(left, right uint64) bool {
	cmp, nan := fcmp64(left, right)
	return cmp < 0 && !nan
}

func sf64IsNaN(value uint64) bool {
	_, _, _, _, nan := funpack64(value)
	return nan
}

func sf64IsInfPos(value uint64) bool {
	fs, fm, _, inf, nan := funpack64(value)
	return inf && !nan && fs == 0 && fm == 0
}

func sf64IsInfNeg(value uint64) bool {
	fs, fm, _, inf, nan := funpack64(value)
	return inf && !nan && fs != 0 && fm == 0
}

func sf64Abs(value uint64) uint64 { return value &^ (1 << 63) }

func sf64ToInt(value uint64) int {
	intValue, ok := f64toint(value)
	if !ok {
		return 0
	}
	return int(intValue)
}

func intTo64(value int) uint64 {
	fs := uint64(0)
	mant := uint64(value)
	if value < 0 {
		fs = 1 << 63
		mant = uint64(-int64(value))
	}
	return fpack64(fs, mant, int(mantbits64), 0)
}

func sf64ExpMulti(hi, lo uint64, exponent int) uint64 {
	reduced := fadd64(hi, sf64Neg(lo))
	reducedSquared := fmul64(reduced, reduced)
	hornerInner := fadd64(expP4, fmul64(reducedSquared, expP5))
	hornerMid := fadd64(expP3, fmul64(reducedSquared, hornerInner))
	hornerOuter := fadd64(expP2, fmul64(reducedSquared, hornerMid))
	horner := fadd64(expP1, fmul64(reducedSquared, hornerOuter))
	coefficient := fadd64(reduced, sf64Neg(fmul64(reducedSquared, horner)))
	twoMinusC := fadd64(two64, sf64Neg(coefficient))
	rcOver := fdiv64(fmul64(reduced, coefficient), twoMinusC)
	inner := fadd64(sf64Neg(lo), rcOver)
	inner = fadd64(inner, sf64Neg(hi))
	value := fadd64(one, sf64Neg(inner))
	return ldexp64(value, exponent)
}

func sf64Exp(value uint64) uint64 {
	if sf64IsNaN(value) {
		return value
	}
	if sf64IsInfPos(value) {
		return inf64
	}
	if sf64IsInfNeg(value) {
		return zero
	}
	if sf64Lt(value, underflow) {
		return zero
	}
	if sf64Lt(overflow, value) {
		return inf64
	}
	if sf64Lt(sf64Abs(value), 0x3E30000000000000) {
		return fadd64(one, value)
	}

	var exponentK int
	if sf64Lt(value, zero) {
		scaled := fadd64(fmul64(log2e, value), sf64Neg(half64))
		exponentK = sf64ToInt(scaled)
	} else {
		scaled := fadd64(fmul64(log2e, value), half64)
		exponentK = sf64ToInt(scaled)
	}

	kFloat := intTo64(exponentK)
	kLn2Hi := fmul64(kFloat, ln2Hi)
	kLn2Lo := fmul64(kFloat, ln2Lo)
	hi := fadd64(value, sf64Neg(kLn2Hi))
	return sf64ExpMulti(hi, kLn2Lo, exponentK)
}

func main() {
	erfArg := -2.7695015832175374
	absValue := math.Float64bits(math.Abs(erfArg))
	truncated := absValue & 0xFFFFFFFF00000000
	zMinusX := fadd64(truncated, sf64Neg(absValue))
	zPlusX := fadd64(truncated, absValue)
	inner := fadd64(fmul64(zMinusX, zPlusX), fdiv64(ra0, sa1)) // stub R/S
	negZSquared := sf64Neg(fmul64(truncated, truncated))
	expPrimary := sf64Exp(fadd64(negZSquared, neg05625))
	expSecondary := sf64Exp(inner)
	fmt.Println("expPrimary", math.Float64frombits(expPrimary), "expSecondary", math.Float64frombits(expSecondary))
	ratio := fdiv64(fmul64(expPrimary, expSecondary), absValue)
	fmt.Println("ratio", math.Float64frombits(ratio), "isnan", math.IsNaN(math.Float64frombits(ratio)))
	fmt.Println("hw erf", math.Erf(erfArg))
}
