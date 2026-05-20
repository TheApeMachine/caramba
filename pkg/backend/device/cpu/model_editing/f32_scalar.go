package model_editing

/*
WeightGraftAddFloat32Scalar adds injection in-place: weights[i] += injection[i].
*/
func WeightGraftAddFloat32Scalar(weights, injection []float32) {
	for index := range weights {
		weights[index] += injection[index]
	}
}
