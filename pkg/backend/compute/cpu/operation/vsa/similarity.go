package vsa

/*
Similarity computes the dot-product cosine similarity between two VSA hypervectors.
Assumes both vectors are already L2-normalised (which Bundle guarantees), so the
dot product equals the cosine similarity directly.
shape=[N], data[0]=a, data[1]=b → out=[dot_product].
*/
type Similarity struct{}

/*
NewSimilarity instantiates a new Similarity operation.
*/
func NewSimilarity() *Similarity { return &Similarity{} }

/*
Forward returns a length-1 slice containing the dot product of data[0] and data[1].
*/
func (similarity *Similarity) Forward(shape []int, data ...[]float64) []float64 {
	return []float64{applyDot(data[0], data[1])}
}
