//go:build cgo && xla

package xla

// #include "optimizer.h"
import "C"

import (
	"fmt"
	stdmath "math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type Registry struct{}

func NewOptimizerRegistry() Registry { return Registry{} }

func (registry Registry) Adam(*state.Dict) (state.Optimizer, error)     { return &Adam{}, nil }
func (registry Registry) AdamW(*state.Dict) (state.Optimizer, error)    { return &AdamW{}, nil }
func (registry Registry) AdaMax(*state.Dict) (state.Optimizer, error)   { return &AdaMax{}, nil }
func (registry Registry) SGD(*state.Dict) (state.Optimizer, error)      { return &SGD{}, nil }
func (registry Registry) Lion(*state.Dict) (state.Optimizer, error)     { return &Lion{}, nil }
func (registry Registry) RMSProp(*state.Dict) (state.Optimizer, error)  { return &RMSProp{}, nil }
func (registry Registry) Hebbian(*state.Dict) (state.Optimizer, error)  { return &Hebbian{}, nil }
func (registry Registry) Lars(*state.Dict) (state.Optimizer, error)     { return &Lars{}, nil }
func (registry Registry) Lamb(*state.Dict) (state.Optimizer, error)     { return &Lamb{}, nil }
func (registry Registry) AdaGrad(*state.Dict) (state.Optimizer, error)  { return &AdaGrad{}, nil }
func (registry Registry) AdaDelta(*state.Dict) (state.Optimizer, error) { return &AdaDelta{}, nil }
func (registry Registry) LBFGS(*state.Dict) (state.Optimizer, error)    { return &LBFGS{}, nil }

type Adam struct{}

func (adam *Adam) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla adam"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	rc := C.xla_optimizer_adam(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(adamLearningRate(stateDict)), C.double(stateDict.Eps),
	)

	return stateDict, xlaOptimizerError(rc, "adam")
}

type AdamW struct{}

func (adamW *AdamW) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla adamw"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	rc := C.xla_optimizer_adamw(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(adamLearningRate(stateDict)), C.double(stateDict.Eps),
		C.double(stateDict.LR*stateDict.WD),
	)

	return stateDict, xlaOptimizerError(rc, "adamw")
}

type AdaMax struct{}

func (adaMax *AdaMax) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla adamax"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	learningRate := stateDict.LR /
		(1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step)))
	rc := C.xla_optimizer_adamax(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(learningRate), C.double(stateDict.Eps),
	)

	return stateDict, xlaOptimizerError(rc, "adamax")
}

type SGD struct{}

func (sgd *SGD) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla sgd"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.xla_optimizer_sgd(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.WD),
		C.double(stateDict.Momentum), boolInt(stateDict.Nesterov),
	)

	return stateDict, xlaOptimizerError(rc, "sgd")
}

type Lion struct{}

func (lion *Lion) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla lion"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.xla_optimizer_lion(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.Beta1),
		C.double(stateDict.Beta2), C.double(stateDict.WD),
	)

	return stateDict, xlaOptimizerError(rc, "lion")
}

type RMSProp struct{}

func (rmsProp *RMSProp) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla rmsprop"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureBuf()
	if stateDict.Centered {
		stateDict.EnsureGradAvg()
	}
	rc := C.xla_optimizer_rmsprop(
		ptr(stateDict.Out), ptr(stateDict.V), ptr(stateDict.Buf), ptr(stateDict.GradAvg),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.LR), C.double(stateDict.Alpha), C.double(stateDict.Eps),
		C.double(stateDict.Momentum), C.double(stateDict.WD), boolInt(stateDict.Centered),
	)

	return stateDict, xlaOptimizerError(rc, "rmsprop")
}

type Hebbian struct{}

func (hebbian *Hebbian) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla hebbian"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.xla_optimizer_hebbian(
		ptr(stateDict.Out), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.MaxNorm),
	)

	return stateDict, xlaOptimizerError(rc, "hebbian")
}

type Lars struct{}

func (lars *Lars) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla lars"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	rc := C.xla_optimizer_lars(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(stateDict.LR), C.double(stateDict.Eta),
		C.double(stateDict.Momentum), C.double(stateDict.WD), C.double(stateDict.Eps),
	)

	return stateDict, xlaOptimizerError(rc, "lars")
}

type Lamb struct{}

func (lamb *Lamb) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla lamb"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	bc1Inv := 1.0 / (1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step)))
	bc2Inv := 1.0 / (1 - stdmath.Pow(stateDict.Beta2, float64(stateDict.Step)))
	rc := C.xla_optimizer_lamb(
		ptr(stateDict.Out), ptr(stateDict.M), ptr(stateDict.V),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.LR), C.double(stateDict.Beta1), C.double(stateDict.Beta2),
		C.double(stateDict.Eps), C.double(stateDict.WD), C.double(bc1Inv), C.double(bc2Inv),
	)

	return stateDict, xlaOptimizerError(rc, "lamb")
}

type AdaGrad struct{}

func (adaGrad *AdaGrad) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla adagrad"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	learningRate := stateDict.LR
	if stateDict.LRDecay != 0 {
		learningRate /= 1 + float64(stateDict.Step-1)*stateDict.LRDecay
	}
	rc := C.xla_optimizer_adagrad(
		ptr(stateDict.Out), ptr(stateDict.V), ptr(stateDict.Params), ptr(stateDict.Grads),
		C.int(len(stateDict.Params)), C.double(learningRate), C.double(stateDict.Eps),
		C.double(stateDict.WD),
	)

	return stateDict, xlaOptimizerError(rc, "adagrad")
}

type AdaDelta struct{}

func (adaDelta *AdaDelta) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla adadelta"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureBuf()
	rc := C.xla_optimizer_adadelta(
		ptr(stateDict.Out), ptr(stateDict.V), ptr(stateDict.Buf),
		ptr(stateDict.Params), ptr(stateDict.Grads), C.int(len(stateDict.Params)),
		C.double(stateDict.Rho), C.double(stateDict.Eps), C.double(stateDict.WD),
	)

	return stateDict, xlaOptimizerError(rc, "adadelta")
}

type LBFGS struct{}

func (lbfgs *LBFGS) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("xla lbfgs"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureHistory()
	previousParams, previousGrads := previous(stateDict)
	sHistory := flatten(stateDict.SHist, stateDict.HistSize, len(stateDict.Params))
	yHistory := flatten(stateDict.YHist, stateDict.HistSize, len(stateDict.Params))
	head := C.int(stateDict.Head)
	count := C.int(stateDict.Count)
	rc := C.xla_optimizer_lbfgs(
		ptr(stateDict.Out), ptr(sHistory), ptr(yHistory), ptr(stateDict.RhoHist),
		(*C.int)(unsafe.Pointer(&head)), (*C.int)(unsafe.Pointer(&count)),
		ptr(stateDict.Params), ptr(stateDict.Grads), ptr(previousParams), ptr(previousGrads),
		boolInt(stateDict.PrevParams != nil), C.int(len(stateDict.Params)),
		C.int(stateDict.HistSize), C.double(stateDict.LR), boolInt(stateDict.LineSearch),
		C.double(stateDict.C1),
	)
	if err := xlaOptimizerError(rc, "lbfgs"); err != nil {
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

func xlaOptimizerError(rc C.int, name string) error {
	if rc == 0 {
		return nil
	}
	return fmt.Errorf("xla optimizer: %s fused kernel failed", name)
}
