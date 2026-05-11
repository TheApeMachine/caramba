package markov_blanket

import "fmt"

/*
Partition extracts sensory, active, internal, and external state vectors
from a joint state vector x, given boolean masks for each partition.

shape = [N, N_s, N_a, N_i, N_e]
data[0] = x [N]
data[1] = sensory_mask [N]   (1.0 = member)
data[2] = active_mask  [N]
data[3] = internal_mask [N]
data[4] = external_mask [N]

Returns concatenation [x_s | x_a | x_i | x_e].
*/
type Partition struct{}

func NewPartition() *Partition { return &Partition{} }

func (op *Partition) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 5 {
		panic(fmt.Errorf("markov_blanket: Partition: len(shape)=%d, need 5", len(shape)).Error())
	}

	if len(data) < 5 {
		panic(fmt.Errorf("markov_blanket: Partition: len(data)=%d, need 5", len(data)).Error())
	}

	N, Ns, Na, Ni, Ne := shape[0], shape[1], shape[2], shape[3], shape[4]
	x := data[0]

	if len(x) != N {
		panic(fmt.Errorf("markov_blanket: Partition: len(x)=%d, need N=%d", len(x), N).Error())
	}

	out := make([]float64, Ns+Na+Ni+Ne)
	applyPartition(out, x, data[1], data[2], data[3], data[4], Ns, Na, Ni, Ne)

	return out
}

func applyPartitionScalar(
	out, x, smask, amask, imask, emask []float64,
	Ns, Na, Ni, Ne int,
) {
	si, ai, ii, ei := 0, Ns, Ns+Na, Ns+Na+Ni

	for idx, val := range x {
		switch {
		case smask[idx] != 0 && si < Ns:
			out[si] = val
			si++
		case amask[idx] != 0 && ai < Ns+Na:
			out[ai] = val
			ai++
		case imask[idx] != 0 && ii < Ns+Na+Ni:
			out[ii] = val
			ii++
		case emask[idx] != 0 && ei < Ns+Na+Ni+Ne:
			out[ei] = val
			ei++
		}
	}
}
