package runtime

import (
	"context"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

/*
WeightedExecutor extends Executor with external inputs and checkpoint weights.
*/
type WeightedExecutor interface {
	Executor

	ExecuteWithWeights(
		ctx context.Context,
		graph *ir.Graph,
		targets []*ir.Node,
		weightsPath string,
		externalInputs map[string]tensor.Tensor,
	) (map[string]tensor.Tensor, error)
}
