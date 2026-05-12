package vsa

import "fmt"

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
	if len(shape) < 1 {
		panic("vsa: Bind.Forward: len(shape) < 1")
	}

	n := shape[0]

	if n < 0 {
		panic(fmt.Sprintf("vsa: Bind.Forward: shape[0] (n) must be non-negative, got n=%d", n))
	}

	if len(data) < 2 {
		panic(fmt.Sprintf("vsa: Bind.Forward: len(data)=%d, need >= 2", len(data)))
	}

	if len(data[0]) != n || len(data[1]) != n {
		panic(fmt.Sprintf(
			"vsa: Bind.Forward: need len(data[0])==n and len(data[1])==n for n=%d, got %d and %d",
			n, len(data[0]), len(data[1]),
		))
	}

	out := make([]float64, n)
	applyBind(out, data[0], data[1])

	return out
}
