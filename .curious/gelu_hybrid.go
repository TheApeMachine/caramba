//go:build ignore
package main
import ("fmt"; "math"; cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"; "github.com/theapemachine/caramba/pkg/backend/device/cpu/parity")
func erfF(value float32) float32 { /* same as metal */ 
	const erx = 0.8450629115098051
	const one = 1.0
	const small = 1.0 / 268435456.0
	const point84375 = 0.84375
	const point25 = 1.25
	const six = 6.0
	const efx = 0.12837916734433576
	const efx8 = 1.0270333357641007
	sign := value < 0
	absValue := float32(math.Abs(float64(value)))
	if absValue >= six { if sign { return -1 }; return 1 }
	if absValue < point84375 {
		if absValue < small { temp := efx8*absValue + 8*absValue; if sign { return -temp }; return temp }
		z := absValue * absValue
		polyR := efx + z*(-0.3250420967602993+z*(-0.02848174980910479+z*(-0.005770270296489442+z*(-0.000023763016656652608))))
		polyS := one + z*(0.39791722399155353+z*(0.06502224998876729+z*(0.005081306281875766+z*(0.00013249473800432163+z*(-0.000003960228278775368)))))
		ratio := polyR / polyS
		temp := absValue + absValue*ratio
		if sign { return -temp }; return temp
	}
	if absValue < point25 {
		shift := absValue - one
		polyP := -0.0023621185607526594 + shift*0.41485611868374833
		polyQ := one + shift*0.10642088040084423
		frac := polyP / polyQ
		if sign { return -erx - frac }; return erx + frac
	}
	invSq := one / (absValue * absValue)
	polyR := -0.009864944034847148 + invSq*(-0.6938585727071818)
	polyS := one + invSq*19.651271667439257
	z := math.Float32frombits(math.Float32bits(absValue) & 0xFFFFF000)
	expInner := (z-absValue)*(z+absValue) + polyR/polyS
	expTerm := float32(math.Exp(float64(-z*z-0.5625))) * float32(math.Exp(float64(expInner)))
	tail := expTerm / absValue
	if sign { return tail - one }; return one - tail
}
func main() {
	maxULP := 0
	for index := 0; index < 8192; index++ {
		value := float32(1+index%240)/12 - 4
		t := value * 0.7071067811865475
		erf := erfF(t)
		got := float32(0.5 * float64(value) * (1 + float64(erf)))
		ref := cpumath.FastGelu32(value)
		ulp := parity.Float32ULPDistance(ref, got)
		if ulp > maxULP { maxULP = ulp }
	}
	fmt.Println("hybrid max ULP", maxULP)
}
