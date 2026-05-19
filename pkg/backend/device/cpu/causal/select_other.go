//go:build !amd64 && !arm64

package causal

func CateFloat32Native(treated, control, out []float32) {
	cateF32Generic(treated, control, out)
}

func CounterfactualFloat32Native(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	counterfactualF32Generic(out, observedY, observedX, counterfactualX, slope)
}

func DoInterveneFloat32Native(out, adjacency []float32, intervened []int32, nodeCount int) {
	doInterveneF32Generic(out, adjacency, intervened, nodeCount)
}

func BackdoorAdjustmentFloat32Native(
	conditional, marginalZ, out []float32,
	xCount, zCount, yCount int,
) {
	backdoorAdjustmentF32Generic(conditional, marginalZ, out, xCount, zCount, yCount)
}

func FrontdoorAdjustmentFloat32Native(
	mediatorGivenX, outcomeGivenXM, marginalX, out []float32,
	xCount, mCount, yCount int,
) {
	frontdoorAdjustmentF32Generic(mediatorGivenX, outcomeGivenXM, marginalX, out, xCount, mCount, yCount)
}

func IvEstimateFloat32Native(instrument, treatment, outcome []float32) float32 {
	return ivEstimateF32Generic(instrument, treatment, outcome)
}

func MarkovFlowFloat32Native(
	mi []float32,
	partition []int32,
	out []float32,
	nodeCount int,
	targetLabel int32,
) {
	markovFlowF32Generic(mi, partition, out, nodeCount, targetLabel)
}
