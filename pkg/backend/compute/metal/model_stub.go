//go:build !darwin || !cgo

package metal

import cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"

/*
ModelOps stubs the Metal model ops — all methods delegate to the CPU
implementation with CPUMatMul when Metal is unavailable.
*/
type ModelOps struct{}

func NewModelOps() *ModelOps { return &ModelOps{} }

func (metalModel *ModelOps) NewLoader(source, file, cache string) *cpumodel.Loader {
	return cpumodel.NewLoader(source, file, cache)
}

func (metalModel *ModelOps) NewSurgery(
	source, op, at, after, name string, layer []float64,
) *cpumodel.Surgery {
	return cpumodel.NewSurgery(source, op, at, after, name, layer)
}

func (metalModel *ModelOps) NewGraft(source, at, mode string) *cpumodel.Graft {
	return cpumodel.NewGraft(source, at, mode)
}

func (metalModel *ModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRA(source, preset, targets, rank, alpha)
}

func (metalModel *ModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapter(source, at, reduction)
}

func (metalModel *ModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return cpumodel.NewFreeze(source, pattern, except, frozen)
}
