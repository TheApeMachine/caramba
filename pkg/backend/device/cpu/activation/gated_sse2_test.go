//go:build amd64

package activation

import (
	"math"
	"testing"
)

func scalarSwiGLU(gate, up float32) float32 {
	// swiglu(x) = x * sigmoid(x)
	// sigmoid(x) = 1 / (1 + exp(-x))
	sig := float32(1.0 / (1.0 + math.Exp(float64(-gate))))
	return gate * sig * up
}

func scalarLinGLU(gate, up float32) float32 {
	return gate * up
}

func scalarSiGLU(gate, up float32) float32 {
	sig := float32(1.0 / (1.0 + math.Exp(float64(-gate))))
	return sig * up
}

func scalarSeGLU(gate, up float32) float32 {
	sig := float32(1.0 / (1.0 + math.Exp(float64(-gate))))
	return sig * up
}

func TestSwiGLUTensorsF32SSE2(t *testing.T) {
	sizes := []int{1, 7, 64, 1024, 8192}
	for _, n := range sizes {
		gate := make([]float32, n)
		up := make([]float32, n)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			gate[i] = float32(i)*0.1 - 0.5
			up[i] = float32(i)*0.1 + 0.5
		}
		SwiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
		for i := 0; i < n; i++ {
			expected := scalarSwiGLU(gate[i], up[i])
			if math.Abs(float64(dst[i]-expected)) > 1e-4 {
				t.Errorf("SwiGLU mismatch at %d: got %f, expected %f", i, dst[i], expected)
			}
		}
	}
}

func TestLinGLUTensorsF32SSE2(t *testing.T) {
	sizes := []int{1, 7, 64, 1024, 8192}
	for _, n := range sizes {
		gate := make([]float32, n)
		up := make([]float32, n)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			gate[i] = float32(i)*0.1 - 0.5
			up[i] = float32(i)*0.1 + 0.5
		}
		LinGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
		for i := 0; i < n; i++ {
			expected := scalarLinGLU(gate[i], up[i])
			if math.Abs(float64(dst[i]-expected)) > 1e-4 {
				t.Errorf("LinGLU mismatch at %d: got %f, expected %f", i, dst[i], expected)
			}
		}
	}
}

func TestSiGLUTensorsF32SSE2(t *testing.T) {
	sizes := []int{1, 7, 64, 1024, 8192}
	for _, n := range sizes {
		gate := make([]float32, n)
		up := make([]float32, n)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			gate[i] = float32(i)*0.1 - 0.5
			up[i] = float32(i)*0.1 + 0.5
		}
		SiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
		for i := 0; i < n; i++ {
			expected := scalarSiGLU(gate[i], up[i])
			if math.Abs(float64(dst[i]-expected)) > 1e-4 {
				t.Errorf("SiGLU mismatch at %d: got %f, expected %f", i, dst[i], expected)
			}
		}
	}
}

func TestSeGLUTensorsF32SSE2(t *testing.T) {
	sizes := []int{1, 7, 64, 1024, 8192}
	for _, n := range sizes {
		gate := make([]float32, n)
		up := make([]float32, n)
		dst := make([]float32, n)
		for i := 0; i < n; i++ {
			gate[i] = float32(i)*0.1 - 0.5
			up[i] = float32(i)*0.1 + 0.5
		}
		SeGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
		for i := 0; i < n; i++ {
			expected := scalarSeGLU(gate[i], up[i])
			if math.Abs(float64(dst[i]-expected)) > 1e-4 {
				t.Errorf("SeGLU mismatch at %d: got %f, expected %f", i, dst[i], expected)
			}
		}
	}
}

func BenchmarkSwiGLUTensorsF32SSE2(b *testing.B) {
	n := 8192
	gate := make([]float32, n)
	up := make([]float32, n)
	dst := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SwiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
	}
}

func BenchmarkLinGLUTensorsF32SSE2(b *testing.B) {
	n := 8192
	gate := make([]float32, n)
	up := make([]float32, n)
	dst := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LinGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
	}
}

func BenchmarkSiGLUTensorsF32SSE2(b *testing.B) {
	n := 8192
	gate := make([]float32, n)
	up := make([]float32, n)
	dst := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SiGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
	}
}

func BenchmarkSeGLUTensorsF32SSE2(b *testing.B) {
	n := 8192
	gate := make([]float32, n)
	up := make([]float32, n)
	dst := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SeGLUTensorsF32SSE2(&dst[0], &gate[0], &up[0], n)
	}
}
