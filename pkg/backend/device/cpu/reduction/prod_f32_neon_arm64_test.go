//go:build arm64

package reduction

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestReduceProdFloat32NEONParity(t *testing.T) {
	rng := rand.New(rand.NewSource(0x700D))

	for _, count := range parityNs {
		t.Run(fmt.Sprintf("N=%d", count), func(t *testing.T) {
			values := make([]float32, count)

			for index := range values {
				values[index] = float32(0.5 + rng.Float64())
			}

			want := scalarReduceProd(values)
			got := ReduceProdFloat32NEONAsm(&values[0], len(values))

			if got != want && float32ULPDistance(got, want) > 16 {
				t.Fatalf("N=%d got=%g want=%g ulp=%d",
					count, got, want, float32ULPDistance(got, want))
			}
		})
	}
}

func scalarReduceProd(values []float32) float32 {
	product := float64(1)

	for _, value := range values {
		product *= float64(value)
	}

	return float32(product)
}

func float32ULPDistance(left, right float32) int64 {
	leftBits := math.Float32bits(left)
	rightBits := math.Float32bits(right)

	if leftBits > rightBits {
		return int64(leftBits - rightBits)
	}

	return int64(rightBits - leftBits)
}
