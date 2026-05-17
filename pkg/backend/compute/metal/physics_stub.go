//go:build !darwin || !cgo

package metal

import "fmt"

/*
MetalPhysics is stubbed on non-Darwin builds; calls return an "unavailable"
error so consumers can fall back to CPU/CUDA backends explicitly.
*/
type MetalPhysics struct{}

func NewPhysics(metallib string) (*MetalPhysics, error) {
	return nil, fmt.Errorf("metal physics: Metal backend is only available on darwin with cgo")
}

func (metalPhysics *MetalPhysics) Laplacian1D([]float64, int, float64) ([]float64, error) {
	return nil, fmt.Errorf("metal physics: Metal backend not built")
}

func (metalPhysics *MetalPhysics) Laplacian2D([]float64, int, int, float64) ([]float64, error) {
	return nil, fmt.Errorf("metal physics: Metal backend not built")
}

func (metalPhysics *MetalPhysics) Laplacian3D([]float64, int, int, int, float64) ([]float64, error) {
	return nil, fmt.Errorf("metal physics: Metal backend not built")
}

func (metalPhysics *MetalPhysics) Laplacian([]int, float64, []float64) ([]float64, error) {
	return nil, fmt.Errorf("metal physics: Metal backend not built")
}
