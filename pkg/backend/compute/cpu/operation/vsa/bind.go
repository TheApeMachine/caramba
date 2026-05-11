package vsa

/*
Bind computes the VSA binding operation (elementwise multiplication / Hadamard product).
In FHRR-style VSA, binding combines two hypervectors to represent a relationship;
the result is approximately orthogonal to both inputs when vectors are random.
shape=[N], data[0]=a, data[1]=b → out[N].
*/
type Bind struct{}

/*
NewBind instantiates a new Bind operation.
*/
func NewBind() *Bind { return &Bind{} }

/*
Forward computes elementwise product of data[0] and data[1].
*/
func (bind *Bind) Forward(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, n)
	applyBind(out, data[0], data[1])
	return out
}
