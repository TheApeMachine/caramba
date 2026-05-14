//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_vsa.h"
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

var xlaVSAUsers atomic.Int32

/*
XLAVSAOps dispatches VSA (Vector Symbolic Algebra) operations to the XLA runtime via PJRT.
Operations are expressed as StableHLO modules, compiled once and cached for reuse.
*/
type XLAVSAOps struct {
	mu           sync.Mutex
	shutdownOnce sync.Once
	closed       atomic.Bool
}

/*
NewVSAOps initialises the PJRT client for the given platform ("cpu"/"gpu").
*/
func NewVSAOps(platform string) (*XLAVSAOps, error) {
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_vsa_init(cp); rc != 0 {
		return nil, vsaPJRTError("xla_vsa_init", rc)
	}

	xlaVSAUsers.Add(1)

	return &XLAVSAOps{}, nil
}

/*
Shutdown releases all PJRT VSA resources.
*/
func (xlaVSAOps *XLAVSAOps) Shutdown() {
	xlaVSAOps.shutdownOnce.Do(func() {
		if xlaVSAUsers.Add(-1) == 0 {
			C.xla_vsa_shutdown()
		}

		xlaVSAOps.closed.Store(true)
	})
}

func cIntVSA(name string, n int) (C.int, error) {
	if n < 0 || n > math.MaxInt32 {
		return 0, fmt.Errorf("%s: dimension %d out of range for C.int", name, n)
	}

	return C.int(int32(n)), nil
}

func cSignedIntVSA(name string, n int) (C.int, error) {
	if n < math.MinInt32 || n > math.MaxInt32 {
		return 0, fmt.Errorf("%s: value %d out of range for C.int", name, n)
	}

	return C.int(int32(n)), nil
}

func vsaPJRTError(op string, rc C.int) error {
	if msg := strings.TrimSpace(C.GoString(C.xla_vsa_get_last_error())); msg != "" {
		return fmt.Errorf("%s failed (rc=%d): %s", op, rc, msg)
	}

	return fmt.Errorf("%s failed (rc=%d)", op, rc)
}

/*
Bind computes elementwise product of data[0] and data[1] via XLA.
shape=[N].
*/
func (xlaVSAOps *XLAVSAOps) Bind(shape []int, data ...[]float64) ([]float64, error) {
	xlaVSAOps.mu.Lock()
	defer xlaVSAOps.mu.Unlock()

	if xlaVSAOps.closed.Load() {
		return nil, fmt.Errorf("Bind: XLAVSAOps shut down")
	}

	if len(shape) < 1 {
		return nil, fmt.Errorf("Bind: len(shape) < 1")
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf("Bind: n must be > 0")
	}

	cn, err := cIntVSA("Bind.n", n)
	if err != nil {
		return nil, err
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("Bind: len(data) < 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("Bind: data slices shorter than n=%d", n)
	}

	out := make([]float64, n)

	rc := C.xla_vsa_bind(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, vsaPJRTError("xla_vsa_bind", rc)
	}

	return out, nil
}

/*
Bundle superimposes all input vectors and returns an L2-normalised result via XLA.
shape=[N].
*/
func (xlaVSAOps *XLAVSAOps) Bundle(shape []int, data ...[]float64) ([]float64, error) {
	xlaVSAOps.mu.Lock()
	defer xlaVSAOps.mu.Unlock()

	if xlaVSAOps.closed.Load() {
		return nil, fmt.Errorf("Bundle: XLAVSAOps shut down")
	}

	if len(shape) < 1 {
		return nil, fmt.Errorf("Bundle: len(shape) < 1")
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf("Bundle: n must be > 0")
	}

	cn, err := cIntVSA("Bundle.n", n)
	if err != nil {
		return nil, err
	}

	numVecs := len(data)

	if numVecs <= 0 {
		return nil, fmt.Errorf("Bundle: need at least one input vector")
	}

	ptrs := make([]*C.double, numVecs)

	for i, vec := range data {
		if len(vec) < n {
			return nil, fmt.Errorf("Bundle: len(data[%d])=%d, need >= %d", i, len(vec), n)
		}

		ptrs[i] = (*C.double)(unsafe.Pointer(&vec[0]))
	}

	cNumVecs, err := cIntVSA("Bundle.numVecs", numVecs)
	if err != nil {
		return nil, err
	}

	out := make([]float64, n)

	rc := C.xla_vsa_bundle(
		(**C.double)(unsafe.Pointer(&ptrs[0])),
		cNumVecs,
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(ptrs)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, vsaPJRTError("xla_vsa_bundle", rc)
	}

	return out, nil
}

/*
Similarity computes dot-product similarity between data[0] and data[1] via XLA.
shape=[N], returns length-1 slice.
*/
func (xlaVSAOps *XLAVSAOps) Similarity(shape []int, data ...[]float64) ([]float64, error) {
	xlaVSAOps.mu.Lock()
	defer xlaVSAOps.mu.Unlock()

	if xlaVSAOps.closed.Load() {
		return nil, fmt.Errorf("Similarity: XLAVSAOps shut down")
	}

	if len(shape) < 1 {
		return nil, fmt.Errorf("Similarity: len(shape) < 1")
	}

	n := shape[0]

	if n <= 0 {
		return nil, fmt.Errorf("Similarity: n must be > 0")
	}

	cn, err := cIntVSA("Similarity.n", n)
	if err != nil {
		return nil, err
	}

	if len(data) < 2 {
		return nil, fmt.Errorf("Similarity: len(data) < 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("Similarity: data slices shorter than n=%d", n)
	}

	out := make([]float64, 1)

	rc := C.xla_vsa_similarity(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, vsaPJRTError("xla_vsa_similarity", rc)
	}

	return out, nil
}

func (xlaVSAOps *XLAVSAOps) Permute(
	shape []int,
	shift int,
	data ...[]float64,
) ([]float64, error) {
	xlaVSAOps.mu.Lock()
	defer xlaVSAOps.mu.Unlock()

	if xlaVSAOps.closed.Load() {
		return nil, fmt.Errorf("Permute: XLAVSAOps shut down")
	}

	if len(shape) < 1 {
		return nil, fmt.Errorf("Permute: len(shape) < 1")
	}

	n := shape[0]
	if n <= 0 {
		return nil, fmt.Errorf("Permute: n must be > 0")
	}

	cn, err := cIntVSA("Permute.n", n)
	if err != nil {
		return nil, err
	}

	cshift, err := cSignedIntVSA("Permute.shift", shift)
	if err != nil {
		return nil, err
	}

	if len(data) < 1 || len(data[0]) < n {
		return nil, fmt.Errorf("Permute: input slice shorter than n=%d", n)
	}

	out := make([]float64, n)
	rc := C.xla_vsa_permute(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
		cshift,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, vsaPJRTError("xla_vsa_permute", rc)
	}

	return out, nil
}

func (xlaVSAOps *XLAVSAOps) InversePermute(
	shape []int,
	shift int,
	data ...[]float64,
) ([]float64, error) {
	xlaVSAOps.mu.Lock()
	defer xlaVSAOps.mu.Unlock()

	if xlaVSAOps.closed.Load() {
		return nil, fmt.Errorf("InversePermute: XLAVSAOps shut down")
	}

	if len(shape) < 1 {
		return nil, fmt.Errorf("InversePermute: len(shape) < 1")
	}

	n := shape[0]
	if n <= 0 {
		return nil, fmt.Errorf("InversePermute: n must be > 0")
	}

	cn, err := cIntVSA("InversePermute.n", n)
	if err != nil {
		return nil, err
	}

	cshift, err := cSignedIntVSA("InversePermute.shift", shift)
	if err != nil {
		return nil, err
	}

	if len(data) < 1 || len(data[0]) < n {
		return nil, fmt.Errorf("InversePermute: input slice shorter than n=%d", n)
	}

	out := make([]float64, n)
	rc := C.xla_vsa_inverse_permute(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		cn,
		cshift,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, vsaPJRTError("xla_vsa_inverse_permute", rc)
	}

	return out, nil
}
