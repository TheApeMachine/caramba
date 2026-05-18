package metal

const metalReductionThreadCount = 256

type metalReductionOp int

const (
	metalReductionSum metalReductionOp = iota
	metalReductionMean
	metalReductionProd
	metalReductionMin
	metalReductionMax
	metalReductionArgmin
	metalReductionArgmax
	metalReductionL1Norm
	metalReductionL2Norm
	metalReductionVariance
	metalReductionStddev
)

func metalReductionPartialCount(elementCount int) int {
	return (elementCount + metalReductionThreadCount - 1) / metalReductionThreadCount
}
