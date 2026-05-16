//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "optimizer.h"
// int metal_optimizer_init(const char *metallib_path);
// const char *metal_optimizer_last_error(void);
import "C"

import (
	"fmt"
	stdmath "math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type Registry struct {
	metallib string
}

func NewOptimizerRegistry() Registry {
	return NewOptimizerRegistryWithMetallib("")
}

func NewOptimizerRegistryWithMetallib(metallib string) Registry {
	return Registry{metallib: metallib}
}

func (registry Registry) Adam(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &Adam{}, nil
}

func (registry Registry) AdamW(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &AdamW{}, nil
}

func (registry Registry) AdaMax(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &AdaMax{}, nil
}

func (registry Registry) SGD(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &SGD{}, nil
}

func (registry Registry) Lion(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &Lion{}, nil
}

func (registry Registry) RMSProp(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &RMSProp{}, nil
}

func (registry Registry) Hebbian(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &Hebbian{}, nil
}

func (registry Registry) Lars(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &Lars{}, nil
}

func (registry Registry) Lamb(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &Lamb{}, nil
}

func (registry Registry) AdaGrad(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &AdaGrad{}, nil
}

func (registry Registry) AdaDelta(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &AdaDelta{}, nil
}

func (registry Registry) LBFGS(config *state.Dict) (state.Optimizer, error) {
	if err := registry.ensure(config); err != nil {
		return nil, err
	}
	return &LBFGS{}, nil
}

type Adam struct{}

func (adam *Adam) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal adam"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	rc := C.metal_optimizer_adam(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(adamLearningRate(stateDict)), C.double(stateDict.Eps),
	)

	return stateDict, metalOptimizerError(rc, "adam")
}

type AdamW struct{}

func (adamW *AdamW) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal adamw"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	rc := C.metal_optimizer_adamw(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(adamLearningRate(stateDict)), C.double(stateDict.Eps),
		C.double(stateDict.LR*stateDict.WD),
	)

	return stateDict, metalOptimizerError(rc, "adamw")
}

type AdaMax struct{}

func (adaMax *AdaMax) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal adamax"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	learningRate := stateDict.LR /
		(1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step)))
	rc := C.metal_optimizer_adamax(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(learningRate), C.double(stateDict.Eps),
	)

	return stateDict, metalOptimizerError(rc, "adamax")
}

type SGD struct{}

func (sgd *SGD) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal sgd"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.metal_optimizer_sgd(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.WD),
		C.double(stateDict.Momentum), boolInt(stateDict.Nesterov),
	)

	return stateDict, metalOptimizerError(rc, "sgd")
}

type Lion struct{}

func (lion *Lion) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal lion"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.metal_optimizer_lion(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.Beta1),
		C.double(stateDict.Beta2), C.double(stateDict.WD),
	)

	return stateDict, metalOptimizerError(rc, "lion")
}

type RMSProp struct{}

func (rmsProp *RMSProp) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal rmsprop"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureBuf()
	if stateDict.Centered {
		stateDict.EnsureGradAvg()
	}

	rc := C.metal_optimizer_rmsprop(
		ptr(stateDict.Out), ptr(stateDict.V), ptr(stateDict.Buf), ptr(stateDict.GradAvg),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.LR), C.double(stateDict.Alpha), C.double(stateDict.Eps),
		C.double(stateDict.Momentum), C.double(stateDict.WD), boolInt(stateDict.Centered),
	)

	return stateDict, metalOptimizerError(rc, "rmsprop")
}

type Hebbian struct{}

func (hebbian *Hebbian) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal hebbian"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.metal_optimizer_hebbian(
		ptr(stateDict.Out), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.MaxNorm),
	)

	return stateDict, metalOptimizerError(rc, "hebbian")
}

type Lars struct{}

func (lars *Lars) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal lars"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.metal_optimizer_lars(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.Eta),
		C.double(stateDict.Momentum), C.double(stateDict.WD), C.double(stateDict.Eps),
	)

	return stateDict, metalOptimizerError(rc, "lars")
}

type Lamb struct{}

func (lamb *Lamb) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal lamb"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	bc1Inv := 1.0 / (1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step)))
	bc2Inv := 1.0 / (1 - stdmath.Pow(stateDict.Beta2, float64(stateDict.Step)))
	rc := C.metal_optimizer_lamb(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.LR), C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(stateDict.Eps), C.double(stateDict.WD), C.double(bc1Inv), C.double(bc2Inv),
	)

	return stateDict, metalOptimizerError(rc, "lamb")
}

type AdaGrad struct{}

func (adaGrad *AdaGrad) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal adagrad"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	learningRate := stateDict.LR
	if stateDict.LRDecay != 0 {
		learningRate /= 1 + float64(stateDict.Step-1)*stateDict.LRDecay
	}
	rc := C.metal_optimizer_adagrad(
		ptr(stateDict.Out), ptr(stateDict.V), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(learningRate), C.double(stateDict.Eps),
		C.double(stateDict.WD),
	)

	return stateDict, metalOptimizerError(rc, "adagrad")
}

type AdaDelta struct{}

func (adaDelta *AdaDelta) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal adadelta"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureBuf()
	rc := C.metal_optimizer_adadelta(
		ptr(stateDict.Out), ptr(stateDict.V), ptr(stateDict.Buf),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Rho), C.double(stateDict.Eps), C.double(stateDict.WD),
	)

	return stateDict, metalOptimizerError(rc, "adadelta")
}

type LBFGS struct{}

func (lbfgs *LBFGS) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("metal lbfgs"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureHistory()
	previousParams, previousGrads := previous(stateDict)
	sHistory := flatten(stateDict.SHist, stateDict.HistSize, len(stateDict.Params))
	yHistory := flatten(stateDict.YHist, stateDict.HistSize, len(stateDict.Params))
	head := C.int(stateDict.Head)
	count := C.int(stateDict.Count)
	rc := C.metal_optimizer_lbfgs(
		ptr(stateDict.Out), ptr(sHistory), ptr(yHistory), ptr(stateDict.RhoHist),
		(*C.int)(unsafe.Pointer(&head)), (*C.int)(unsafe.Pointer(&count)),
		ptr(stateDict.Params), ptr(stateDict.Grads), ptr(previousParams), ptr(previousGrads),
		boolInt(stateDict.PrevParams != nil), C.int(len(stateDict.Params)),
		C.int(stateDict.HistSize), C.double(stateDict.LR), boolInt(stateDict.LineSearch),
		C.double(stateDict.C1),
	)
	if err := metalOptimizerError(rc, "lbfgs"); err != nil {
		return nil, err
	}

	stateDict.Head = int(head)
	stateDict.Count = int(count)
	expand(stateDict.SHist, sHistory, len(stateDict.Params))
	expand(stateDict.YHist, yHistory, len(stateDict.Params))
	stateDict.PrevParams = clone(stateDict.Params)
	stateDict.PrevGrads = clone(stateDict.Grads)

	return stateDict, nil
}

func adamLearningRate(stateDict *state.Dict) float64 {
	bc1 := 1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step))
	bc2 := 1 - stdmath.Pow(stateDict.Beta2, float64(stateDict.Step))

	return stateDict.LR * stdmath.Sqrt(bc2) / bc1
}

func (registry Registry) ensure(config *state.Dict) error {
	metallib := registry.metallib
	if metallib == "" {
		metallib = metalLibrary(config, "optimizer.metallib")
	}
	path := C.CString(metallib)
	defer C.free(unsafe.Pointer(path))

	return metalOptimizerError(C.metal_optimizer_init(path), "init")
}

func previous(stateDict *state.Dict) ([]float64, []float64) {
	if stateDict.PrevParams != nil && stateDict.PrevGrads != nil {
		return stateDict.PrevParams, stateDict.PrevGrads
	}

	return make([]float64, len(stateDict.Params)), make([]float64, len(stateDict.Params))
}

func flatten(history [][]float64, historySize, count int) []float64 {
	flat := make([]float64, historySize*count)
	for historyIndex, values := range history {
		copy(flat[historyIndex*count:(historyIndex+1)*count], values)
	}
	return flat
}

func expand(history [][]float64, flat []float64, count int) {
	for historyIndex := range history {
		if history[historyIndex] == nil || len(history[historyIndex]) != count {
			history[historyIndex] = make([]float64, count)
		}
		copy(history[historyIndex], flat[historyIndex*count:(historyIndex+1)*count])
	}
}

func clone(values []float64) []float64 {
	clonedValues := make([]float64, len(values))
	copy(clonedValues, values)
	return clonedValues
}

func ptr(values []float64) *C.double {
	if len(values) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&values[0]))
}

func boolInt(value bool) C.int {
	if value {
		return 1
	}
	return 0
}

func metalOptimizerError(rc C.int, name string) error {
	if rc == 0 {
		return nil
	}

	message := C.GoString(C.metal_optimizer_last_error())
	if message != "" {
		return fmt.Errorf("metal optimizer: %s fused kernel failed: %s", name, message)
	}

	return fmt.Errorf("metal optimizer: %s fused kernel failed", name)
}
