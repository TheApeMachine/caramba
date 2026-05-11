//go:build !linux || !cgo || !cuda

package cuda

import cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"

/*
CUDAModelOps stubs the CUDA model ops — all methods delegate to the CPU
implementation with CPUMatMul when CUDA is unavailable.
*/
type CUDAModelOps struct{}

func NewCUDAModelOps() *CUDAModelOps { return &CUDAModelOps{} }

func (cudaModel *CUDAModelOps) NewLoader(source, file, cache string) *cpumodel.Loader {
	return cpumodel.NewLoader(source, file, cache)
}

func (cudaModel *CUDAModelOps) NewSurgery(
	source, op, at, after, name string, layer []float64,
) *cpumodel.Surgery {
	return cpumodel.NewSurgery(source, op, at, after, name, layer)
}

func (cudaModel *CUDAModelOps) NewGraft(source, at, mode string) *cpumodel.Graft {
	return cpumodel.NewGraft(source, at, mode)
}

func (cudaModel *CUDAModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRA(source, preset, targets, rank, alpha)
}

func (cudaModel *CUDAModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapter(source, at, reduction)
}

func (cudaModel *CUDAModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return cpumodel.NewFreeze(source, pattern, except, frozen)
}
