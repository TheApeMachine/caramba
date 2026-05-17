package diffusion

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
weightDispatch routes the runtime bridge's per-call WeightBinder hook
to the correct weight store for each graph the program calls. The
diffusion program has three distinct graphs (text_encoder, denoiser,
vae) that each need their own safetensors.
*/
type weightDispatch struct {
	stores map[string]*modelweights.Store
}

func newWeightDispatch(stores map[string]*modelweights.Store) backend.WeightBinder {
	dispatch := weightDispatch{stores: stores}

	return dispatch.bind
}

func (dispatch weightDispatch) bind(irGraph *ir.Graph, module program.GraphModule) error {
	store, ok := dispatch.stores[module.ID]

	if !ok || store == nil {
		return fmt.Errorf("diffusion/runtime: no weight store registered for graph %q", module.ID)
	}

	return modelweights.BindIR(irGraph, store)
}
