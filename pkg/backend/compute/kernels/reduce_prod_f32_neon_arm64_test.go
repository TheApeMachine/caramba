//go:build arm64

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestReduceProdFloat32NEONParity(t *testing.T) {
	rng := rand.New(rand.NewSource(0x700D))

	for _, n := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			values := make([]float32, n)

			for index := range values {
				values[index] = float32(0.5 + rng.Float64())
			}

			want := scalarReduceProd(values)
			got := reduceProdFloat32NEONAsm(&values[0], len(values))

			if got != want && float32ULPDistance(got, want) > 16 {
				t.Fatalf("got=%g want=%g ulp=%d", got, want, float32ULPDistance(got, want))
			}
		})
	}
}

func TestAddFloat64NEONParity(t *testing.T) {
	rng := rand.New(rand.NewSource(0xADD064))

	for _, n := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := make([]float64, n)
			right := make([]float64, n)
			got := make([]float64, n)
			want := make([]float64, n)

			for index := range left {
				left[index] = rng.NormFloat64()
				right[index] = rng.NormFloat64()
				want[index] = left[index] + right[index]
			}

			addFloat64Native(got, left, right)

			for index := range got {
				if got[index] != want[index] {
					t.Fatalf("lane %d got=%g want=%g", index, got[index], want[index])
				}
			}
		})
	}
}

func TestMatMulFloat64NEONParity(t *testing.T) {
	cases := []struct {
		rows, inner, cols int
	}{
		{1, 1, 2}, {3, 7, 8}, {16, 32, 64},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("%dx%dx%d", testCase.rows, testCase.inner, testCase.cols)

		t.Run(label, func(t *testing.T) {
			left := randFloat64Matrix(testCase.rows, testCase.inner, 0x7001)
			right := randFloat64Matrix(testCase.inner, testCase.cols, 0x7002)
			got := make([]float64, testCase.rows*testCase.cols)
			want := make([]float64, testCase.rows*testCase.cols)

			matmulFloat64Native(got, left, right, testCase.rows, testCase.inner, testCase.cols)
			scalarMatMulFloat64(want, left, right, testCase.rows, testCase.inner, testCase.cols)

			for index := range got {
				diff := math.Abs(got[index] - want[index])
				if diff > 1e-12 {
					t.Fatalf("lane %d got=%g want=%g", index, got[index], want[index])
				}
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

func scalarMatMulFloat64(out, left, right []float64, rows, inner, cols int) {
	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := left[rowIndex*inner+innerIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				out[rowIndex*cols+colIndex] +=
					leftValue * right[innerIndex*cols+colIndex]
			}
		}
	}
}

func randFloat64Matrix(rows, cols int, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	matrix := make([]float64, rows*cols)

	for index := range matrix {
		matrix[index] = rng.NormFloat64() * 0.1
	}

	return matrix
}
