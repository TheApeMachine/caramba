package compute

import "github.com/theapemachine/caramba/pkg/backend/compute/ir"

/*
OptimizerContract declares optimizer operation IDs used by train graph nodes.
*/
type OptimizerContract struct {
	operationIDs []ir.OpType
}

/*
StandardOptimizerContract is the default optimizer capability contract. It is
initialized from ir.TrainOptimizerOperationIDs and can be used directly by
standard backends; callers that need a different optimizer surface can construct
their own OptimizerContract.
*/
var StandardOptimizerContract = OptimizerContract{
	operationIDs: ir.TrainOptimizerOperationIDs(),
}

/*
OperationIDs returns a copy of the contract's optimizer operation IDs as
ir.OpType values so callers cannot mutate the internal slice.
*/
func (optimizerContract OptimizerContract) OperationIDs() []ir.OpType {
	return append([]ir.OpType(nil), optimizerContract.operationIDs...)
}
