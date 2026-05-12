//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "markov_blanket.h"
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

/*
MetalMarkovBlanket dispatches Markov blanket operations on the GPU (markov_blanket.metallib).

The underlying C/ObjC layer uses process-wide init state; the Go mutex serialises use of
this wrapper from multiple goroutines.
*/
type MetalMarkovBlanket struct {
	mu       sync.Mutex
	metallib string
}

func NewMarkovBlanket(metallib string) (*MetalMarkovBlanket, error) {
	if strings.TrimSpace(metallib) == "" {
		return nil, fmt.Errorf("NewMarkovBlanket: metallib path is empty")
	}

	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_mb_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_mb_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalMarkovBlanket{metallib: metallib}, nil
}

/*
Close releases reference state for this module (idempotent on the C side).
*/
func (op *MetalMarkovBlanket) Close() error {
	op.mu.Lock()
	defer op.mu.Unlock()

	if rc := C.metal_mb_cleanup(); rc != 0 {
		return fmt.Errorf("metal_mb_cleanup failed (rc=%d)", rc)
	}

	return nil
}

// Partition extracts state partitions. shape=[N,Ns,Na,Ni,Ne], data[0]=x [N], data[1]=masks [4N].
func (op *MetalMarkovBlanket) Partition(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 5 {
		return nil, fmt.Errorf("MetalMarkovBlanket.Partition: need len(shape) >= 5")
	}

	N, Ns, Na, Ni, Ne := shape[0], shape[1], shape[2], shape[3], shape[4]

	if N <= 0 || Ns < 0 || Na < 0 || Ni < 0 || Ne < 0 {
		return nil, fmt.Errorf("MetalMarkovBlanket.Partition: invalid shape")
	}

	outLen := Ns + Na + Ni + Ne

	if outLen <= 0 {
		return nil, fmt.Errorf("MetalMarkovBlanket.Partition: output length must be > 0")
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalMarkovBlanket.Partition: need len(data) >= 2")
	}

	if len(data[0]) != N {
		return nil, fmt.Errorf("MetalMarkovBlanket.Partition: len(x) must be %d", N)
	}

	if len(data[1]) != 4*N {
		return nil, fmt.Errorf("MetalMarkovBlanket.Partition: len(masks) must be %d", 4*N)
	}

	x := toFloat32(data[0])
	masks := toFloat32(data[1])
	out := make([]float32, outLen)

	rc := C.metal_mb_partition(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&masks[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(N), C.int(Ns), C.int(Na), C.int(Ni), C.int(Ne),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_partition failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

// FlowInternal: out[Ni] = W[Ni*Ns] @ x_sens[Ns] + bias[Ni]
func (op *MetalMarkovBlanket) FlowInternal(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowInternal: need len(shape) >= 2")
	}

	Ni, Ns := shape[0], shape[1]

	if Ni <= 0 || Ns <= 0 {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowInternal: need Ni>0 and Ns>0")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowInternal: need len(data) >= 3")
	}

	if len(data[0]) < Ns || len(data[1]) < Ni*Ns || len(data[2]) < Ni {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowInternal: slice length mismatch")
	}

	xSens := toFloat32(data[0])
	w := toFloat32(data[1])
	bias := toFloat32(data[2])
	out := make([]float32, Ni)

	rc := C.metal_mb_flow_internal(
		(*C.float)(unsafe.Pointer(&xSens[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&bias[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(Ni), C.int(Ns),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_flow_internal failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

// FlowActive: out[Na] = W[Na*Ni] @ x_int[Ni] + bias[Na]
func (op *MetalMarkovBlanket) FlowActive(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowActive: need len(shape) >= 2")
	}

	Na, Ni := shape[0], shape[1]

	if Na <= 0 || Ni <= 0 {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowActive: need Na>0 and Ni>0")
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowActive: need len(data) >= 3")
	}

	if len(data[0]) < Ni || len(data[1]) < Na*Ni || len(data[2]) < Na {
		return nil, fmt.Errorf("MetalMarkovBlanket.FlowActive: slice length mismatch")
	}

	xInt := toFloat32(data[0])
	w := toFloat32(data[1])
	bias := toFloat32(data[2])
	out := make([]float32, Na)

	rc := C.metal_mb_flow_active(
		(*C.float)(unsafe.Pointer(&xInt[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&bias[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(Na), C.int(Ni),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_flow_active failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

// MutualInformation: shape=[N,M], data[0]=X[T*N], data[1]=Y[T*M]. Returns scalar.
func (op *MetalMarkovBlanket) MutualInformation(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalMarkovBlanket.MutualInformation: need len(shape) >= 2")
	}

	n, m := shape[0], shape[1]

	if n <= 0 || m <= 0 {
		return nil, fmt.Errorf("MetalMarkovBlanket.MutualInformation: need N>0 and M>0")
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalMarkovBlanket.MutualInformation: need len(data) >= 2")
	}

	x0 := data[0]
	y0 := data[1]

	if len(x0)%n != 0 {
		return nil, fmt.Errorf("MetalMarkovBlanket.MutualInformation: len(X) not divisible by N=%d", n)
	}

	t := len(x0) / n

	if t < 2 {
		return nil, fmt.Errorf("MetalMarkovBlanket.MutualInformation: need T>=2 samples")
	}

	if len(y0) != t*m {
		return nil, fmt.Errorf("MetalMarkovBlanket.MutualInformation: len(Y) must be T*M=%d", t*m)
	}

	xData := toFloat32(x0)
	yData := toFloat32(y0)
	out := make([]float32, 1)

	rc := C.metal_mb_mutual_information(
		(*C.float)(unsafe.Pointer(&xData[0])),
		(*C.float)(unsafe.Pointer(&yData[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(t), C.int(n), C.int(m),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_mutual_information failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}
