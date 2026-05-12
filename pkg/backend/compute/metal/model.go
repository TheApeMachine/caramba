//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"
	"unsafe"

	cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"
)

/*
ModelOps exposes model surgery/graft/lora/adapter/freeze operations,
wiring the math-intensive matmul paths to the Metal kernel.
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

/*
NewLoRA injects metalMatMul so the B·A computation runs on the GPU.
*/
func (metalModel *ModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRAWithMatMul(source, preset, targets, rank, alpha, metalMatMul)
}

/*
NewAdapter injects metalMatMul so the bottleneck forward pass runs on the GPU.
*/
func (metalModel *ModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapterWithMatMul(source, at, reduction, metalMatMul)
}

func (metalModel *ModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return cpumodel.NewFreeze(source, pattern, except, frozen)
}

/*
metalMatMul dispatches a row-major matmul to the GPU via metal_matmul.
a [M×K], b [K×N] → c [M×N], all row-major float64 converted to float32.
Note: float64→float32→float64 conversion incurs precision loss (~7 decimal digits).
*/
func metalMatMul(a, b []float64, M, K, N int) ([]float64, error) {
	if M < 0 || K < 0 || N < 0 {
		return nil, fmt.Errorf("metal: metalMatMul requires M, K, N >= 0")
	}

	if M == 0 || N == 0 {
		return []float64{}, nil
	}

	if len(a) < M*K || len(b) < K*N {
		return nil, fmt.Errorf("metal: metalMatMul slice lengths too short for given dimensions")
	}

	aF32 := toFloat32(a)
	bF32 := toFloat32(b)
	cF32 := make([]float32, M*N)

	rc := C.metal_matmul(
		(*C.float)(unsafe.Pointer(&aF32[0])),
		(*C.float)(unsafe.Pointer(&bF32[0])),
		(*C.float)(unsafe.Pointer(&cF32[0])),
		C.int(M), C.int(K), C.int(N),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_matmul failed (rc=%d) for M=%d K=%d N=%d", rc, M, K, N)
	}

	return toFloat64(cF32), nil
}
