package model_editing

/*
WeightGraftAddFloat32Native adds an injection vector into weights (model.graft read_write).
*/
func WeightGraftAddFloat32Native(weights, injection []float32) {
	elementCount := len(weights)

	if elementCount == 0 {
		return
	}

	weightGraftAddFloat32Kernel(weights, injection, elementCount)
}
