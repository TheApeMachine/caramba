//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_math.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"

	cpumodel "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"
)

/*
XLAModelOps exposes model surgery/graft/lora/adapter/freeze operations,
wiring the math-intensive matmul paths to the XLA PJRT kernel.
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

/*
NewLoRA injects xlaMatMul so the B·A computation runs via XLA PJRT.
*/
func (xlaModel *XLAModelOps) NewLoRA(
	source, preset string, targets []string, rank int, alpha float64,
) *cpumodel.LoRA {
	return cpumodel.NewLoRAWithMatMul(source, preset, targets, rank, alpha, xlaMatMul)
}

/*
NewAdapter injects xlaMatMul so the bottleneck forward pass runs via XLA PJRT.
*/
func (xlaModel *XLAModelOps) NewAdapter(source, at string, reduction int) *cpumodel.Adapter {
	return cpumodel.NewAdapterWithMatMul(source, at, reduction, xlaMatMul)
}

func (xlaModel *XLAModelOps) NewFreeze(source, pattern, except string, frozen bool) *cpumodel.Freeze {
	return cpumodel.NewFreeze(source, pattern, except, frozen)
}

/*
xlaMatMul dispatches a row-major matmul via xla_matmul (PJRT C API).
a [M×K], b [K×N] → c [M×N], all row-major float64 (double).
*/
func xlaMatMul(a, b []float64, M, K, N int) ([]float64, error) {
	if M <= 0 || K <= 0 || N <= 0 {
		return nil, fmt.Errorf("xla: xlaMatMul requires M, K, N > 0")
	}

	if M != 0 && N > math.MaxInt/M {
		return nil, fmt.Errorf("xla: xlaMatMul: M×N overflows int")
	}

	needA := int64(M) * int64(K)
	needB := int64(K) * int64(N)

	if int64(len(a)) < needA {
		return nil, fmt.Errorf(
			"xla: xlaMatMul: len(a)=%d < M×K=%d×%d=%d",
			len(a), M, K, needA,
		)
	}

	if int64(len(b)) < needB {
		return nil, fmt.Errorf(
			"xla: xlaMatMul: len(b)=%d < K×N=%d×%d=%d",
			len(b), K, N, needB,
		)
	}

	c := make([]float64, M*N)

	rc := C.xla_matmul(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(M), C.int(K), C.int(N),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_matmul failed (rc=%d) for dims M=%d K=%d N=%d", rc, M, K, N)
	}

	return c, nil
}
