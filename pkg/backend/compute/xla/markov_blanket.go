//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_markov_blanket.h"
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

/*
XLAMarkovBlanket dispatches Markov blanket operations to XLA via PJRT.
*/
type XLAMarkovBlanket struct {
	shutdownOnce sync.Once
}

func finalizeXLAMarkovBlanket(op *XLAMarkovBlanket) {
	_ = op.Shutdown()
}

func NewMarkovBlanket(platform string) (*XLAMarkovBlanket, error) {
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_mb_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_mb_init failed for platform %q (rc=%d)", config.Platform, rc)
	}

	op := &XLAMarkovBlanket{}
	runtime.SetFinalizer(op, finalizeXLAMarkovBlanket)

	return op, nil
}

// Shutdown releases Markov blanket PJRT resources. It is idempotent and safe to call from a finalizer.
func (op *XLAMarkovBlanket) Shutdown() error {
	var shutdownErr error

	op.shutdownOnce.Do(func() {
		if rc := C.xla_mb_shutdown(); rc != 0 {
			shutdownErr = fmt.Errorf("xla_mb_shutdown failed (rc=%d)", rc)
		}
	})

	return shutdownErr
}

func cIntMB(name string, v int) (C.int, error) {
	if v < 0 || v > math.MaxInt32 {
		return 0, fmt.Errorf("%s: value %d out of range for C.int", name, v)
	}

	return C.int(int32(v)), nil
}

// Partition: shape=[N,Ns,Na,Ni,Ne], data[0]=x, data[1]=masks[4*N].
func (op *XLAMarkovBlanket) Partition(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 5 {
		return nil, fmt.Errorf("Partition: len(shape) < 5")
	}

	N, Ns, Na, Ni, Ne := shape[0], shape[1], shape[2], shape[3], shape[4]

	if N < 0 || Ns < 0 || Na < 0 || Ni < 0 || Ne < 0 {
		return nil, fmt.Errorf("Partition: negative dimension")
	}

	sum := int64(Ns) + int64(Na) + int64(Ni) + int64(Ne)
	if sum > int64(math.MaxInt) || sum < 0 {
		return nil, fmt.Errorf("Partition: output size overflow")
	}

	outLen := int(sum)

	if len(data) < 2 {
		return nil, fmt.Errorf("Partition: len(data) < 2")
	}

	if len(data[0]) < N || len(data[1]) < 4*N {
		return nil, fmt.Errorf("Partition: data slice length mismatch for N=%d", N)
	}

	cN, err := cIntMB("Partition.N", N)
	if err != nil {
		return nil, err
	}

	cNs, err := cIntMB("Partition.Ns", Ns)
	if err != nil {
		return nil, err
	}

	cNa, err := cIntMB("Partition.Na", Na)
	if err != nil {
		return nil, err
	}

	cNi, err := cIntMB("Partition.Ni", Ni)
	if err != nil {
		return nil, err
	}

	cNe, err := cIntMB("Partition.Ne", Ne)
	if err != nil {
		return nil, err
	}

	out := make([]float64, outLen)
	rc := C.xla_mb_partition(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cN, cNs, cNa, cNi, cNe,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_mb_partition failed (rc=%d)", rc)
	}

	return out, nil
}

// FlowInternal: shape=[Ni,Ns], data[0]=x_sens, data[1]=W, data[2]=bias.
func (op *XLAMarkovBlanket) FlowInternal(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("FlowInternal: len(shape) != 2")
	}

	Ni, Ns := shape[0], shape[1]

	if Ni < 0 || Ns < 0 {
		return nil, fmt.Errorf("FlowInternal: negative Ni or Ns")
	}

	if len(data) != 3 {
		return nil, fmt.Errorf("FlowInternal: len(data) != 3")
	}

	if len(data[0]) < Ns || len(data[1]) < Ni*Ns || len(data[2]) < Ni {
		return nil, fmt.Errorf("FlowInternal: data slice length mismatch for Ni=%d Ns=%d", Ni, Ns)
	}

	cNi, err := cIntMB("FlowInternal.Ni", Ni)
	if err != nil {
		return nil, err
	}

	cNs, err := cIntMB("FlowInternal.Ns", Ns)
	if err != nil {
		return nil, err
	}

	out := make([]float64, Ni)
	rc := C.xla_mb_flow_internal(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cNi, cNs,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_mb_flow_internal failed (rc=%d)", rc)
	}

	return out, nil
}

// FlowActive: shape=[Na,Ni], data[0]=x_int, data[1]=W, data[2]=bias.
func (op *XLAMarkovBlanket) FlowActive(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("FlowActive: len(shape) != 2")
	}

	Na, Ni := shape[0], shape[1]

	if Na < 0 || Ni < 0 {
		return nil, fmt.Errorf("FlowActive: negative Na or Ni")
	}

	if len(data) != 3 {
		return nil, fmt.Errorf("FlowActive: len(data) != 3")
	}

	if len(data[0]) < Ni || len(data[1]) < Na*Ni || len(data[2]) < Na {
		return nil, fmt.Errorf("FlowActive: data slice length mismatch for Na=%d Ni=%d", Na, Ni)
	}

	cNa, err := cIntMB("FlowActive.Na", Na)
	if err != nil {
		return nil, err
	}

	cNi, err := cIntMB("FlowActive.Ni", Ni)
	if err != nil {
		return nil, err
	}

	out := make([]float64, Na)
	rc := C.xla_mb_flow_active(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cNa, cNi,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_mb_flow_active failed (rc=%d)", rc)
	}

	return out, nil
}

// MutualInformation: shape=[N,M], data[0]=X[T*N], data[1]=Y[T*M].
func (op *XLAMarkovBlanket) MutualInformation(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("MutualInformation: len(shape) < 2")
	}

	N, M := shape[0], shape[1]

	if N <= 0 || M <= 0 || N > math.MaxInt32 || M > math.MaxInt32 {
		return nil, fmt.Errorf("MutualInformation: invalid N=%d M=%d", N, M)
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("MutualInformation: len(data) < 2")
	}

	if len(data[0])%N != 0 {
		return nil, fmt.Errorf("MutualInformation: len(data[0])=%d not a multiple of N=%d", len(data[0]), N)
	}

	T := len(data[0]) / N

	if T <= 0 {
		return nil, fmt.Errorf("MutualInformation: non-positive T from len(data[0])=%d N=%d", len(data[0]), N)
	}

	if len(data[1]) < T*M {
		return nil, fmt.Errorf("MutualInformation: len(data[1])=%d, need T*M=%d*%d", len(data[1]), T, M)
	}

	cT, err := cIntMB("MutualInformation.T", T)
	if err != nil {
		return nil, err
	}

	cN, err := cIntMB("MutualInformation.N", N)
	if err != nil {
		return nil, err
	}

	cM, err := cIntMB("MutualInformation.M", M)
	if err != nil {
		return nil, err
	}

	out := make([]float64, 1)
	rc := C.xla_mb_mutual_information(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cT, cN, cM,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_mb_mutual_information failed (rc=%d)", rc)
	}

	return out, nil
}
