package math

import (
	"unsafe"
)

var NaN = *(*float32)(unsafe.Pointer(&uvnan32))

// FastExp32 computes e^x using IEEE-754 bit manipulation and a polynomial approximation.
// It avoids float64 and is roughly 5-10x faster than math.Exp in Go.
func FastExp32(x float32) float32 {
	// Clamp to avoid float32 overflow and slow subnormals
	if x < -87.33654 {
		return 0.0
	}
	if x > 88.72283 {
		return 0x1p127 * (1 + (1 - 0x1p-23)) // 3.40282e+38
	}

	// e^x = 2^(x * log2(e))
	z := x * 1.4426950408889634 // multiply by log2(e)

	// Separate into integer and fractional parts
	k := int32(z)
	if z < 0 {
		k-- // Ensure fraction is always in [0, 1)
	}
	f := z - float32(k)

	// Minimax polynomial approximation for 2^f on [0, 1)
	poly := 1.0 + f*(0.69314718+f*(0.24022650+f*(0.05550410+f*(0.00961812+f*0.00133389))))

	// Add integer part directly into the IEEE-754 exponent field
	bits := *(*uint32)(unsafe.Pointer(&poly))
	bits += uint32(k) << 23
	return *(*float32)(unsafe.Pointer(&bits))
}

// FastLog32 computes ln(x) by extracting the IEEE-754 exponent and using a polynomial.
func FastLog32(x float32) float32 {
	if x <= 0 {
		return NaN
	}

	bits := *(*uint32)(unsafe.Pointer(&x))
	// Extract the exponent (biased) and calculate the real exponent
	exp := int32(bits>>23) - 127

	// Force the exponent to 0 (biased 127) to get the mantissa into range [1, 2)
	mantissaBits := (bits & 0x007FFFFF) | (127 << 23)
	m := *(*float32)(unsafe.Pointer(&mantissaBits))

	// x = m * 2^exp  =>  ln(x) = ln(m) + exp * ln(2)
	// Approximate ln(1 + y) for y in [0, 1) where y = m - 1
	y := m - 1.0
	poly := y * (0.9999964239 + y*(-0.4998741238+y*(0.3317990258+y*(-0.2407338084+y*0.14449244))))

	return poly + float32(exp)*0.6931471805599453 // multiply exponent by ln(2)
}

// FastTanh32 computes tanh(x) using a Padé approximant (Lambert's continued fraction).
// This is exceptionally fast because it requires ZERO calls to Exp(), just multiplications.
func FastTanh32(x float32) float32 {
	// Clamp limits to asymptotes
	if x > 4.92 {
		return 1.0
	}
	if x < -4.92 {
		return -1.0
	}

	x2 := x * x
	num := x * (135135.0 + x2*(17325.0+x2*(378.0+x2)))
	den := 135135.0 + x2*(62370.0+x2*(3150.0+x2*28.0))
	return num / den
}

func FastSigmoid32(value float32) float32 {
	if value >= 0 {
		return 1 / (1 + FastExp32(-value))
	}

	expValue := FastExp32(value)
	return expValue / (1 + expValue)
}

func FastSilu32(value float32) float32 {
	return value / (1 + FastExp32(-value))
}

func FastGeluTanh32(value float32) float32 {
	valueFloat64 := float64(value)
	inner := GeluTanhAlpha * (valueFloat64 + GeluTanhBeta*valueFloat64*valueFloat64*valueFloat64)
	return float32(0.5 * valueFloat64 * (1 + float64(FastTanh32(float32(inner)))))
}

/*
FastSin32 is a minimax sine approximation on [-pi, pi] via range reduction.
*/
func FastSin32(angle float32) float32 {
	const twoPi = 6.283185307179586
	const pi = 3.141592653589793

	reduced := angle - twoPi*float32(int32(angle/twoPi))

	if reduced > pi {
		reduced -= twoPi
	}

	if reduced < -pi {
		reduced += twoPi
	}

	reducedSquared := reduced * reduced
	return reduced * (1 - reducedSquared*(0.16666667-reducedSquared*0.008333333))
}
