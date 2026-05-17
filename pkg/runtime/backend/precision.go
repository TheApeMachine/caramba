package backend

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
applyDefaultPrecision sets Precision on every IR node that does not
already declare one. Pure nodes (math ops) get the default; impure
or input nodes whose ValueType is unset still get the default for
their compute precision but keep their storage DType intact. This
is the mechanism Metal and CUDA backends use to execute a manifest
authored at float64 without rewriting every node config.
*/
func applyDefaultPrecision(graph *ir.Graph, precision tensor.DType) {
	if precision == "" {
		return
	}

	for _, node := range graph.Nodes() {
		current := node.ValueType()

		if current.Precision == precision {
			continue
		}

		updated := current
		updated.Precision = precision
		node.SetValueType(updated)
	}
}
