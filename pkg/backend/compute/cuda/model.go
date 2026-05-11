//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "math.h"
import "C"

import (
	"unsafe"

	cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"
)

/*
CUDAModelOps exposes model surgery/graft/lora/adapter/freeze operations,
wiring the math-intensive matmul paths to the CUDA kernel.
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

/*
NewLoRA injects cudaMatMul so the B·A computation runs on the GPU.
*/
func (cudaModel *CUDAModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRAWithMatMul(source, preset, targets, rank, alpha, cudaMatMul)
}

/*
NewAdapter injects cudaMatMul so the bottleneck forward pass runs on the GPU.
*/
func (cudaModel *CUDAModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapterWithMatMul(source, at, reduction, cudaMatMul)
}

func (cudaModel *CUDAModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return cpumodel.NewFreeze(source, pattern, except, frozen)
}

/*
cudaMatMul dispatches a row-major matmul to the GPU via cuda_matmul.
a [M×K], b [K×N] → c [M×N], all row-major float64 (double).
*/
func cudaMatMul(a, b []float64, M, K, N int) []float64 {
	c := make([]float64, M*N)

	C.cuda_matmul(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(M), C.int(K), C.int(N),
	)

	return c
}
