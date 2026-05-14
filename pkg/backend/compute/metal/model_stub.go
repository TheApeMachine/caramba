//go:build !darwin || !cgo

package metal

import cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"

/*
ModelOps stubs the Metal model ops when Metal is not built in.
*/
type ModelOps struct{}

func NewModelOps() *ModelOps { return &ModelOps{} }

func (metalModel *ModelOps) NewLoader(source, file, cache string) *cpumodel.Loader {
	return &cpumodel.Loader{}
}

func (metalModel *ModelOps) NewSurgery(
	source, op, at, after, name string, layer []float64,
) *cpumodel.Surgery {
	return &cpumodel.Surgery{}
}

func (metalModel *ModelOps) NewGraft(source, at, mode string) *cpumodel.Graft {
	return &cpumodel.Graft{}
}

func (metalModel *ModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRAWithMatMul(source, preset, targets, rank, alpha, metalUnavailableMatMul)
}

func (metalModel *ModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapterWithMatMul(source, at, reduction, metalUnavailableMatMul)
}

func (metalModel *ModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return &cpumodel.Freeze{}
}

func metalUnavailableMatMul(a, b []float64, M, K, N int) ([]float64, error) {
	return nil, metalUnavailable()
}
