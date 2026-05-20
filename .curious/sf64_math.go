//go:build ignore

package main

import "math"

// Softfloat64 math.Exp + math.Erf (Go math package algorithms).

const (
	ln2Hi = 0x3FE62E42FEE00000
	ln2Lo = 0x3DEA39EF35793C76
	log2e = 0x3FF71547652B82FE
	overflow  = 0x40862E42FEFA39EF
	underflow = 0xC0874910D52D5AF7
	half64    = 0x3FE0000000000000
	sqrt2Inv  = 0x3FE6A09E667F3BCD
	expP1     = 0x3FC5555555555555
	expP2     = 0xBF66C16C16BEBD93
	expP3     = 0x3F11566AAF25DE2C
	expP4     = 0xBEBBBD41C5D26BF1
	expP5     = 0x3E66376972BEA4D0
	two64     = 0x4000000000000000
)

func ldexp64(value uint64, exponent int) uint64 {
	fs, fm, fe, fi, fn := funpack64(value)
	if fi || fn {
		return value
	}
	return fpack64(fs, fm, fe+exponent, 0)
}

func sf64ToInt(value uint64) int {
	intValue, ok := f64toint(value)
	if !ok {
		return 0
	}
	return int(intValue)
}

func sf64ExpMulti(hi, lo uint64, exponent int) uint64 {
	reduced := fadd64(hi, sf64Neg(lo))
	reducedSquared := fmul64(reduced, reduced)
	horner := fadd64(expP1, fmul64(reducedSquared, fadd64(expP2, fmul64(reducedSquared, fadd64(expP3, fmul64(reducedSquared, fadd64(expP4, fmul64(reducedSquared, expP5)))))))
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
	absValue := sf64Abs(value)
	if sf64Lt(absValue, small) {
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

	kFloat := ldexp64(one, exponentK)
	kLn2Hi := fmul64(kFloat, ln2Hi)
	kLn2Lo := fmul64(kFloat, ln2Lo)
	hi := fadd64(value, sf64Neg(kLn2Hi))
	return sf64ExpMulti(hi, kLn2Lo, exponentK)
}

func sf64Erf(value uint64) uint64 {
	if sf64IsNaN(value) {
		return nan64
	}
	if sf64IsInfPos(value) {
		return one
	}
	if sf64IsInfNeg(value) {
		return sf64Neg(one)
	}

	sign := false
	absValue := value
	if sf64Lt(value, zero) {
		absValue = sf64Neg(value)
		sign = true
	}

	if sf64Lt(absValue, p84375) {
		var temp uint64
		if sf64Lt(absValue, small) {
			if sf64Lt(absValue, veryTiny) {
				eightX := fmul64(eight, absValue)
				efx8X := fmul64(efx8, absValue)
				temp = fmul64(eighth, fadd64(eightX, efx8X))
			} else {
				temp = fadd64(absValue, fmul64(efx, absValue))
			}
		} else {
			z := fmul64(absValue, absValue)
			polyR := fadd64(pp0, fmul64(z, fadd64(pp1, fmul64(z, fadd64(pp2, fmul64(z, fadd64(pp3, fmul64(z, pp4)))))))
			polyS := fadd64(one, fmul64(z, fadd64(qq1, fmul64(z, fadd64(qq2, fmul64(z, fadd64(qq3, fmul64(z, fadd64(qq4, fmul64(z, qq5)))))))))
			ratio := fdiv64(polyR, polyS)
			temp = fadd64(absValue, fmul64(absValue, ratio))
		}
		if sign {
			return sf64Neg(temp)
		}
		return temp
	}

	if sf64Lt(absValue, p25) {
		shift := fadd64(absValue, sf64Neg(one))
		polyP := fadd64(pa0, fmul64(shift, fadd64(pa1, fmul64(shift, fadd64(pa2, fmul64(shift, fadd64(pa3, fmul64(shift, fadd64(pa4, fmul64(shift, fadd64(pa5, fmul64(shift, pa6)))))))))))
		polyQ := fadd64(one, fmul64(shift, fadd64(qa1, fmul64(shift, fadd64(qa2, fmul64(shift, fadd64(qa3, fmul64(shift, fadd64(qa4, fmul64(shift, fadd64(qa5, fmul64(shift, qa6)))))))))))
		quotient := fdiv64(polyP, polyQ)
		if sign {
			return fadd64(sf64Neg(erx), sf64Neg(quotient))
		}
		return fadd64(erx, quotient)
	}

	if !sf64Lt(absValue, six) {
		if sign {
			return sf64Neg(one)
		}
		return one
	}

	invSquare := fdiv64(one, fmul64(absValue, absValue))
	var polyR, polyS uint64
	if sf64Lt(absValue, inv035) {
		polyR = fadd64(ra0, fmul64(invSquare, fadd64(ra1, fmul64(invSquare, fadd64(ra2, fmul64(invSquare, fadd64(ra3, fmul64(invSquare, fadd64(ra4, fmul64(invSquare, fadd64(ra5, fmul64(invSquare, fadd64(ra6, fmul64(invSquare, ra7)))))))))))))
		polyS = fadd64(one, fmul64(invSquare, fadd64(sa1, fmul64(invSquare, fadd64(sa2, fmul64(invSquare, fadd64(sa3, fmul64(invSquare, fadd64(sa4, fmul64(invSquare, fadd64(sa5, fmul64(invSquare, fadd64(sa6, fmul64(invSquare, fadd64(sa7, fmul64(invSquare, sa8))))))))))))))
	} else {
		polyR = fadd64(rb0, fmul64(invSquare, fadd64(rb1, fmul64(invSquare, fadd64(rb2, fmul64(invSquare, fadd64(rb3, fmul64(invSquare, fadd64(rb4, fmul64(invSquare, fadd64(rb5, fmul64(invSquare, rb6))))))))))
		polyS = fadd64(one, fmul64(invSquare, fadd64(sb1, fmul64(invSquare, fadd64(sb2, fmul64(invSquare, fadd64(sb3, fmul64(invSquare, fadd64(sb4, fmul64(invSquare, fadd64(sb5, fmul64(invSquare, fadd64(sb6, fmul64(invSquare, sb7))))))))))))
	}

	truncated := absValue & 0xFFFFFFFF00000000
	zMinusX := fadd64(truncated, sf64Neg(absValue))
	zPlusX := fadd64(truncated, absValue)
	inner := fadd64(fmul64(zMinusX, zPlusX), fdiv64(polyR, polyS))
	negZSquared := sf64Neg(fmul64(truncated, truncated))
	expPrimary := sf64Exp(fadd64(negZSquared, neg05625))
	expSecondary := sf64Exp(inner)
	ratio := fdiv64(fmul64(expPrimary, expSecondary), absValue)
	if sign {
		return fadd64(ratio, sf64Neg(one))
	}
	return fadd64(one, sf64Neg(ratio))
}

func geluSoftfloat32Full(value float32) float32 {
	value64 := f32to64(math.Float32bits(value))
	erfArg := fmul64(value64, sqrt2Inv)
	erfVal := sf64Erf(erfArg)
	onePlus := fadd64(one, erfVal)
	prod := fmul64(half64, fmul64(value64, onePlus))
	return math.Float32frombits(f64to32(prod))
}
