package compute

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func validateDevicePrecision(device *Device, graph *ir.Graph) error {
	if device == nil || graph == nil {
		return nil
	}

	if device.id.Location != tensor.Metal {
		return nil
	}

	for _, node := range graph.Nodes() {
		valueType := node.ValueType()

		if valueType.DType != dtype.Float64 {
			continue
		}

		if valueType.Precision == dtype.Float32 {
			continue
		}

		return fmt.Errorf(
			"compute: node %q requires F64 precision on %s; set precision: float32 to opt in",
			node.ID(),
			device.id.Location,
		)
	}

	return nil
}
