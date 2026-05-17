//go:build !cgo || !xla

package xla

import "fmt"

/*
XLAPhysicsOps is stubbed when the xla build tag or cgo are not enabled;
calls return an "unavailable" error so consumers can fall back explicitly.
*/
type XLAPhysicsOps struct{}

func NewPhysics(platform string) (*XLAPhysicsOps, error) {
	return nil, fmt.Errorf("xla physics: XLA backend is only available with -tags xla and cgo enabled")
}

func (xlaPhysics *XLAPhysicsOps) Shutdown() {}

func (xlaPhysics *XLAPhysicsOps) Laplacian1D([]float64, int, float64) ([]float64, error) {
	return nil, fmt.Errorf("xla physics: XLA backend not built")
}

func (xlaPhysics *XLAPhysicsOps) Laplacian2D([]float64, int, int, float64) ([]float64, error) {
	return nil, fmt.Errorf("xla physics: XLA backend not built")
}

func (xlaPhysics *XLAPhysicsOps) Laplacian3D([]float64, int, int, int, float64) ([]float64, error) {
	return nil, fmt.Errorf("xla physics: XLA backend not built")
}

func (xlaPhysics *XLAPhysicsOps) Laplacian([]int, float64, []float64) ([]float64, error) {
	return nil, fmt.Errorf("xla physics: XLA backend not built")
}
