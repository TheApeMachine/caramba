//go:build ignore

package main

import (
	"fmt"
	"math"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

// sf64_exp_main.go + sf64_runtime.go in same package via go run *.go

func geluFull(value float32) float32 {
	value64 := f32to64(math.Float32bits(value))
	erfArg := fmul64(value64, sqrt2Inv)
	erfVal := sf64Erf(erfArg)
	onePlus := fadd64(one, erfVal)
	prod := fmul64(half64, fmul64(value64, onePlus))
	return math.Float32frombits(f64to32(prod))
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
				temp = fmul64(eighth, fadd64(fmul64(eight, absValue), fmul64(efx8, absValue)))
			} else {
				temp = fadd64(absValue, fmul64(efx, absValue))
			}
		} else {
			z := fmul64(absValue, absValue)
			polyR := fadd64(pp0, fmul64(z, fadd64(pp1, fmul64(z, fadd64(pp2, fmul64(z, fadd64(pp3, fmul64(z, pp4)))))))
			polyS := fadd64(one, fmul64(z, fadd64(qq1, fmul64(z, fadd64(qq2, fmul64(z, fadd64(qq3, fmul64(z, fadd64(qq4, fmul64(z, qq5)))))))))
			ratio := fmotion64(polyR, polyS)
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
		q := fmotion64(polyP, polyQ)
		if sign {
			return fadd64(sf64Neg(erx), sf64Neg(q))
		}
		return fadd64(erx, q)
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
	ratio := fmotion64(fmul64(expPrimary, expSecondary), absValue)
	if sign {
		return fadd64(ratio, sf64Neg(one))
	}
	return fadd64(one, sf64Neg(ratio))
}

func main() {
	const (
		erx = 0x3FEB0AC160000000
		efx = 0x3FC06EBA8214DB69
		efx8 = 0x3FF06EBA8214DB69
		pp0 = 0x3FC06EBA8214DB68
		pp1 = 0xBFD4CD7D691CB913
		pp2 = 0xBF9D2A51DBD7194F
		pp3 = 0xBF77A291236668E4
		pp4 = 0xBEF8EAD6120016AC
		qq1 = 0x3FD97779CDDADC09
		qq2 = 0x3FB0A54C5536CEBA
		qq3 = 0x3F74D022C4D36B0F
		qq4 = 0x3F215DC9221C1A10
		qq5 = 0xBED09C4342A26120
		pa0 = 0xBF6359B8BEF77538
		pa1 = 0x3FDA8D00AD92B34D
		pa2 = 0xBFD7D240FBB8C3F1
		pa3 = 0x3FD45FCA805120E4
		pa4 = 0xBFBC63983D3E28EC
		pa5 = 0x3FA22A36599795EB
		pa6 = 0xBF61BF380A96073F
		qa1 = 0x3FBB3E6618EEE323
		qa2 = 0x3FE14AF092EB6F33
		qa3 = 0x3FB2635CD99FE9A7
		qa4 = 0x3FC02660E763351F
		qa5 = 0x3F8BEDC26B51DD1C
		qa6 = 0x3F888B545735151D
		ra0 = 0xBF843412600D6435
		ra1 = 0xBFE63416E4BA7360
		ra2 = 0xC0251E0441B0E726
		ra3 = 0xC04F300AE4CBA38D
		ra4 = 0xC0644CB184282266
		ra5 = 0xC067135CEBCCABB2
		ra6 = 0xC054526557E4D2F2
		ra7 = 0xC023A0EFC69AC25C
		sa1 = 0x4033A6B9BD707687
		sa2 = 0x4061350C526AE721
		sa3 = 0x407B290DD58A1A71
		sa4 = 0x40842B1921EC2868
		sa5 = 0x407AD02157700314
		sa6 = 0x405B28A3EE48AE2C
		sa7 = 0x401A47EF8E484A93
		sa8 = 0xBFAEEFF2EE749A62
		rb0 = 0xBF84341239E86F4A
		rb1 = 0xBFE993BA70C285DE
		rb2 = 0xC031C209555F995A
		rb3 = 0xC064145D43C5ED98
		rb4 = 0xC083EC881375F228
		rb5 = 0xC09004616A2E5992
		rb6 = 0xC07E384E9BDC383F
		sb1 = 0x403E568B261D5190
		sb2 = 0x40745CAE221B9F0A
		sb3 = 0x409802EB189D5118
		sb4 = 0x40A8FFB7688C246A
		sb5 = 0x40A3F219CEDF3BE6
		sb6 = 0x407DA874E79FE763
		sb7 = 0xC03670E242712D62
		veryTiny = 0x0080000000000000
		small = 0x3E30000000000000
		six = 0x4018000000000000
		one = 0x3FF0000000000000
		p84375 = 0x3FEB000000000000
		p25 = 0x3FF4000000000000
		inv035 = 0x4006DB6DB6DB6DB7
		neg05625 = 0xBFE2000000000000
		eighth = 0x3FC0000000000000
		eight = 0x4020000000000000
		zero = 0
	)
	_ = erx
	x := float32(1.0/12.0 - 4.0)
	ref := cpumath.FastGelu32(x)
	got := geluFull(x)
	fmt.Printf("ref=%g got=%g ulp=%d isnan=%v\n", ref, got, parity.Float32ULPDistance(ref, got), math.IsNaN(float64(got)))
}
