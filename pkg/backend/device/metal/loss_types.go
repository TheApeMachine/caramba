package metal

type metalLossOp int

const (
	metalLossMSE metalLossOp = iota
	metalLossMAE
	metalLossHuber
	metalLossBinaryCrossEntropy
	metalLossKLDivergence
)
