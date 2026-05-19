package math

import (
	"unsafe"
)

var NaN64 = *(*float64)(unsafe.Pointer(&uvnan))

func FastExp64(x float64) float64 {
	// Clamp to avoid float64 overflow and slow subnormals
	if x < -708.3964185322641 {
		return 0.0
	}
	if x > 709.782712893384 {
		return 0x1p1023 * (1 + (1 - 0x1p-52)) // 1.7976931348623157e+308
	}

	z := x * 0.6931471805599453 // multiply by ln(2)

	// Separate into integer and fractional parts
	k := int64(z)
	if z < 0 {
		k-- // Ensure fraction is always in [0, 1)
	}
	f := z - float64(k)

	// Minimax polynomial approximation for 2^f on [0, 1)
	poly := 1.0 + f*(0.6931471805599453+f*(0.2402265069591007+f*(0.05550410706910451+f*(0.009618127167766223+f*0.001333899727232439))))

	// Add integer part directly into the IEEE-754 exponent field
	bits := *(*uint64)(unsafe.Pointer(&poly))
	bits += uint64(k) << 52
	return *(*float64)(unsafe.Pointer(&bits))
}

func FastLog64(x float64) float64 {
	if x <= 0 {
		return NaN64
	}

	bits := *(*uint64)(unsafe.Pointer(&x))
	// Extract the exponent (biased) and calculate the real exponent
	exp := int64(bits>>52) - 1023

	// Force the exponent to 0 (biased 1023) to get the mantissa into range [1, 2)
	mantissaBits := (bits & 0x000FFFFFFFFFFFFF) | (1023 << 52)
	m := *(*float64)(unsafe.Pointer(&mantissaBits))

	// x = m * 2^exp  =>  ln(x) = ln(m) + exp * ln(2)
	// Approximate ln(1 + y) for y in [0, 1) where y = m - 1
	y := m - 1.0
	poly := y * (0.9999964239 + y*(-0.4998741238+y*(0.3317990258+y*(-0.2407338084+y*0.14449244))))

	return poly + float64(exp)*0.6931471805599453 // multiply exponent by ln(2)
}

func FastTanh64(x float64) float64 {
	// Clamp limits to asymptotes
	if x > 4.92 {
		return 1.0
	}
	if x < -4.92 {
		return -1.0
	}

	x2 := x * x
	num := x * (135135.0 + x2*(17325.0+x2*(378.0+x2)))
	den := 135135.0 + x2*(62370.0+x2*(31185.0+x2*(7335.0+x2*(693.0+x2*(13.0+x2)))))
	return num / den
}

func FastSinh64(x float64) float64 {
	// Clamp to avoid float64 overflow and slow subnormals
	if x < -708.3964185322641 {
		return 0.0
	}
	if x > 709.782712893384 {
		return 0x1p1023 * (1 + (1 - 0x1p-52)) // 1.7976931348623157e+308
	}

	z := x * 0.6931471805599453 // multiply by ln(2)

	// Separate into integer and fractional parts

	k := int64(z)
	if z < 0 {
		k-- // Ensure fraction is always in [0, 1)
	}
	f := z - float64(k)

	// Minimax polynomial approximation for 2^f on [0, 1)
	poly := 1.0 + f*(0.6931471805599453+f*(0.2402265069591007+f*(0.05550410706910451+f*(0.009618127167766223+f*0.001333899727232439))))

	// Add integer part directly into the IEEE-754 exponent field
	bits := *(*uint64)(unsafe.Pointer(&poly))
	bits += uint64(k) << 52
	return *(*float64)(unsafe.Pointer(&bits))
}

func FastSigmoid64(value float64) float64 {
	if value >= 0 {
		return 1 / (1 + FastExp64(-value))
	}

	expValue := FastExp64(value)
	return expValue / (1 + expValue)
}

func FastSilu64(value float64) float64 {
	return value / (1 + FastExp64(-value))
}

func FastGeluTanh64(value float64) float64 {
	inner := GeluTanhAlpha * (value + GeluTanhBeta*value*value*value)
	return 0.5 * value * (1 + FastTanh64(inner))
}
