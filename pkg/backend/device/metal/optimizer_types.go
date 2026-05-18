package metal

type metalOptimizerOp int

const (
	metalOptimizerAdam metalOptimizerOp = iota
	metalOptimizerAdamW
	metalOptimizerAdamax
	metalOptimizerAdagrad
	metalOptimizerRMSprop
	metalOptimizerLion
	metalOptimizerSGD
	metalOptimizerLBFGS
)
