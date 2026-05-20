package metal

import (
	"math"
	"testing"

	cpumath "github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestGeluReferenceProbeIndexZero(t *testing.T) {
	input := extendedUnaryInputValue(0, "gelu")
	reference := cpumath.FastGelu32(input)

	valueFloat64 := float64(input)
	erfArg := valueFloat64 * cpumath.SqrtTwoOverTwo
	erfValue := math.Erf(erfArg)
	product := 0.5 * valueFloat64 * (1 + erfValue)
	emulated := float32(product)

	distance := parity.Float32ULPDistance(reference, emulated)
	t.Logf("input=%g ref=%g emulated=%g ulp=%d erf=%g", input, reference, emulated, distance, erfValue)

	if distance > 2 {
		t.Fatalf("float64 math.Erf product ULP %d > 2", distance)
	}
}
