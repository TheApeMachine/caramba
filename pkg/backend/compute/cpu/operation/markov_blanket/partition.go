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

	smask, amask, imask, emask := data[1], data[2], data[3], data[4]

	for _, label := range []struct {
		name string
		mask []float64
	}{
		{"sensory_mask", smask},
		{"active_mask", amask},
		{"internal_mask", imask},
		{"external_mask", emask},
	} {
		if len(label.mask) != N {
			panic(fmt.Errorf(
				"markov_blanket: Partition: len(%s)=%d, need N=%d",
				label.name, len(label.mask), N,
			).Error())
		}
	}

	for idx := range x {
		set := 0

		if smask[idx] != 0 {
			set++
		}

		if amask[idx] != 0 {
			set++
		}

		if imask[idx] != 0 {
			set++
		}

		if emask[idx] != 0 {
			set++
		}

		if set > 1 {
			panic(fmt.Errorf(
				"markov_blanket: Partition: index %d has %d partition mask(s) set; at most one of sensory/active/internal/external may be non-zero",
				idx, set,
			).Error())
		}
	}

	out := make([]float64, Ns+Na+Ni+Ne)
	applyPartitionScalar(out, x, smask, amask, imask, emask, Ns, Na, Ni, Ne)

	return out
}

// applyPartitionScalar fills out with values from x selected by masks.
// If an index has multiple masks set, Forward rejects before calling here; if this helper is
// called directly with overlaps, the switch order is sensory > active > internal > external.
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
