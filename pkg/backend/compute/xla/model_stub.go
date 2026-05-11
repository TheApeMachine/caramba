//go:build !cgo || !xla

package xla

import cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"

/*
XLAModelOps stubs the XLA model ops — all methods delegate to the CPU
implementation with CPUMatMul when XLA is unavailable.
*/
type XLAModelOps struct{}

func NewXLAModelOps() *XLAModelOps { return &XLAModelOps{} }

func (xlaModel *XLAModelOps) NewLoader(source, file, cache string) *cpumodel.Loader {
	return cpumodel.NewLoader(source, file, cache)
}

func (xlaModel *XLAModelOps) NewSurgery(
	source, op, at, after, name string, layer []float64,
) *cpumodel.Surgery {
	return cpumodel.NewSurgery(source, op, at, after, name, layer)
}

func (xlaModel *XLAModelOps) NewGraft(source, at, mode string) *cpumodel.Graft {
	return cpumodel.NewGraft(source, at, mode)
}

func (xlaModel *XLAModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRA(source, preset, targets, rank, alpha)
}

func (xlaModel *XLAModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapter(source, at, reduction)
}

func (xlaModel *XLAModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return cpumodel.NewFreeze(source, pattern, except, frozen)
}
