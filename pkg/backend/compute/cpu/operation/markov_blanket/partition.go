package markov_blanket

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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

func (partition *Partition) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 5 {
		return nil, fmt.Errorf("markov_blanket.partition: len(shape)=%d, need 5", len(shape))
	}

	if err := stateDict.RequireOperationInputs("markov_blanket.partition", 5); err != nil {
		return nil, err
	}

	dimension := shape[0]
	sensoryCount := shape[1]
	activeCount := shape[2]
	internalCount := shape[3]
	externalCount := shape[4]
	input := stateDict.Inputs[0]

	if len(input) != dimension {
		return nil, fmt.Errorf(
			"markov_blanket.partition: len(x)=%d, need N=%d",
			len(input), dimension,
		)
	}

	sensoryMask := stateDict.Inputs[1]
	activeMask := stateDict.Inputs[2]
	internalMask := stateDict.Inputs[3]
	externalMask := stateDict.Inputs[4]

	for _, label := range []struct {
		name string
		mask []float64
	}{
		{"sensory_mask", sensoryMask},
		{"active_mask", activeMask},
		{"internal_mask", internalMask},
		{"external_mask", externalMask},
	} {
		if len(label.mask) != dimension {
			return nil, fmt.Errorf(
				"markov_blanket.partition: len(%s)=%d, need N=%d",
				label.name, len(label.mask), dimension,
			)
		}
	}

	for index := range input {
		set := 0

		if sensoryMask[index] != 0 {
			set++
		}

		if activeMask[index] != 0 {
			set++
		}

		if internalMask[index] != 0 {
			set++
		}

		if externalMask[index] != 0 {
			set++
		}

		if set > 1 {
			return nil, fmt.Errorf(
				"markov_blanket.partition: index %d has %d partition mask(s) set",
				index, set,
			)
		}
	}

	stateDict.EnsureOperationOutLen(sensoryCount + activeCount + internalCount + externalCount)
	applyPartitionScalar(
		stateDict.Out,
		input,
		sensoryMask,
		activeMask,
		internalMask,
		externalMask,
		sensoryCount,
		activeCount,
		internalCount,
		externalCount,
	)

	return stateDict, nil
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
