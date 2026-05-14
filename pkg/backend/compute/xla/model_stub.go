//go:build !cgo || !xla

package xla

import cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"

/*
XLAModelOps stubs the XLA model ops when XLA is not built in.
*/
type XLAModelOps struct{}

func NewXLAModelOps() *XLAModelOps { return &XLAModelOps{} }

func (xlaModel *XLAModelOps) NewLoader(source, file, cache string) *cpumodel.Loader {
	return &cpumodel.Loader{}
}

func (xlaModel *XLAModelOps) NewSurgery(
	source, op, at, after, name string, layer []float64,
) *cpumodel.Surgery {
	return &cpumodel.Surgery{}
}

func (xlaModel *XLAModelOps) NewGraft(source, at, mode string) *cpumodel.Graft {
	return &cpumodel.Graft{}
}

func (xlaModel *XLAModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRAWithMatMul(source, preset, targets, rank, alpha, xlaUnavailableMatMul)
}

func (xlaModel *XLAModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapterWithMatMul(source, at, reduction, xlaUnavailableMatMul)
}

func (xlaModel *XLAModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return &cpumodel.Freeze{}
}

func xlaUnavailableMatMul(a, b []float64, M, K, N int) ([]float64, error) {
	return nil, xlaOptimizerUnavailable()
}
