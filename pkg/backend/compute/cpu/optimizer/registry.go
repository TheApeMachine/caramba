package optimizer

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/adagrad"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/adam"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/hebbian"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lars"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lbfgs"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lion"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/rmsprop"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/sgd"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type Registry struct{}

func NewOptimizerRegistry() Registry {
	return Registry{}
}

func (registry Registry) Adam(*state.Dict) (state.Optimizer, error) {
	return adam.NewAdam(), nil
}

func (registry Registry) AdamW(*state.Dict) (state.Optimizer, error) {
	return adam.NewAdamW(), nil
}

func (registry Registry) AdaMax(*state.Dict) (state.Optimizer, error) {
	return adam.NewAdaMax(), nil
}

func (registry Registry) SGD(*state.Dict) (state.Optimizer, error) {
	return sgd.NewSGD(), nil
}

func (registry Registry) Lion(*state.Dict) (state.Optimizer, error) {
	return lion.NewLion(), nil
}

func (registry Registry) RMSProp(*state.Dict) (state.Optimizer, error) {
	return rmsprop.NewRMSProp(), nil
}

func (registry Registry) Hebbian(*state.Dict) (state.Optimizer, error) {
	return hebbian.NewHebbian(), nil
}

func (registry Registry) Lars(*state.Dict) (state.Optimizer, error) {
	return lars.NewLARS(), nil
}

func (registry Registry) Lamb(config *state.Dict) (state.Optimizer, error) {
	return lars.NewLAMB(), nil
}

func (registry Registry) AdaGrad(*state.Dict) (state.Optimizer, error) {
	return adagrad.NewAdaGrad(), nil
}

func (registry Registry) AdaDelta(config *state.Dict) (state.Optimizer, error) {
	return adagrad.NewAdaDelta(), nil
}

func (registry Registry) LBFGS(*state.Dict) (state.Optimizer, error) {
	return lbfgs.NewLBFGS(), nil
}
