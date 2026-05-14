//go:build !linux || !cgo || !cuda

package cuda

import cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"

/*
CUDAModelOps stubs the CUDA model ops when CUDA is not built in.
*/
type CUDAModelOps struct{}

func NewCUDAModelOps() *CUDAModelOps { return &CUDAModelOps{} }

func (cudaModel *CUDAModelOps) NewLoader(source, file, cache string) *cpumodel.Loader {
	return &cpumodel.Loader{}
}

func (cudaModel *CUDAModelOps) NewSurgery(
	source, op, at, after, name string, layer []float64,
) *cpumodel.Surgery {
	return &cpumodel.Surgery{}
}

func (cudaModel *CUDAModelOps) NewGraft(source, at, mode string) *cpumodel.Graft {
	return &cpumodel.Graft{}
}

func (cudaModel *CUDAModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRAWithMatMul(source, preset, targets, rank, alpha, cudaUnavailableMatMul)
}

func (cudaModel *CUDAModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapterWithMatMul(source, at, reduction, cudaUnavailableMatMul)
}

func (cudaModel *CUDAModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return &cpumodel.Freeze{}
}

func cudaUnavailableMatMul(a, b []float64, M, K, N int) ([]float64, error) {
	return nil, cudaOptimizerUnavailable()
}
