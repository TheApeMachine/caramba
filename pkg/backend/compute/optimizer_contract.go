package compute

import "github.com/theapemachine/caramba/pkg/backend/compute/ir"

/*
OptimizerContract declares optimizer operation IDs used by train graph nodes.
*/
type OptimizerContract struct {
	operationIDs []ir.OpType
}

var StandardOptimizerContract = OptimizerContract{
	operationIDs: ir.TrainOptimizerOperationIDs(),
}

func (optimizerContract OptimizerContract) OperationIDs() []ir.OpType {
	return append([]ir.OpType(nil), optimizerContract.operationIDs...)
}
