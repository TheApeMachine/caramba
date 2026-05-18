package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Embedding lookup. Args: (table [vocab, hidden], indices [N],
output [N, hidden]). Standard token-embedding primitive.
*/

func init() {
	Default.Register(Kernel{
		Name: "embedding_lookup",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runEmbeddingLookup,
	})

	Default.Register(Kernel{
		Name: "embedding_bag",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runEmbeddingBag,
	})

	// bf16/fp16 embedding lookup — table and output share the reduced
	// dtype; indices stay int32. Pure memcpy at the byte level (each
	// entry is 2 bytes regardless), so the runner is essentially the
	// f32 version with the dtype views swapped.
	Default.Register(Kernel{
		Name: "embedding_lookup",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.Int32},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runEmbeddingLookupBFloat16,
	})

	Default.Register(Kernel{
		Name: "embedding_lookup",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runEmbeddingLookupFloat16,
	})

	Default.Register(Kernel{
		Name: "embedding_bag",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.Int32, dtype.Int32},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runEmbeddingBagBFloat16,
	})

	Default.Register(Kernel{
		Name: "embedding_bag",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Int32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runEmbeddingBagFloat16,
	})
}

func runEmbeddingLookupBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	table, err := args[0].BFloat16Native()
	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()
	if err != nil {
		return err
	}

	out, err := args[2].BFloat16Native()
	if err != nil {
		return err
	}

	tableDims := args[0].Shape().Dims()

	if len(tableDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	vocab := tableDims[0]
	hidden := tableDims[1]

	if len(out) != len(indices)*hidden {
		return tensor.ErrShapeMismatch
	}

	for resultIndex, tokenID := range indices {
		if int(tokenID) < 0 || int(tokenID) >= vocab {
			return tensor.ErrShapeMismatch
		}

		copy(
			out[resultIndex*hidden:(resultIndex+1)*hidden],
			table[int(tokenID)*hidden:(int(tokenID)+1)*hidden],
		)
	}

	return nil
}

func runEmbeddingLookupFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	table, err := args[0].Float16Native()
	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()
	if err != nil {
		return err
	}

	out, err := args[2].Float16Native()
	if err != nil {
		return err
	}

	tableDims := args[0].Shape().Dims()

	if len(tableDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	vocab := tableDims[0]
	hidden := tableDims[1]

	if len(out) != len(indices)*hidden {
		return tensor.ErrShapeMismatch
	}

	for resultIndex, tokenID := range indices {
		if int(tokenID) < 0 || int(tokenID) >= vocab {
			return tensor.ErrShapeMismatch
		}

		copy(
			out[resultIndex*hidden:(resultIndex+1)*hidden],
			table[int(tokenID)*hidden:(int(tokenID)+1)*hidden],
		)
	}

	return nil
}

func runEmbeddingBagBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	table, err := args[0].BFloat16Native()
	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()
	if err != nil {
		return err
	}

	offsets, err := args[2].Int32Native()
	if err != nil {
		return err
	}

	out, err := args[3].BFloat16Native()
	if err != nil {
		return err
	}

	tableDims := args[0].Shape().Dims()

	if len(tableDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	hidden := tableDims[1]
	bags := len(offsets)

	if len(out) != bags*hidden {
		return tensor.ErrShapeMismatch
	}

	// f32 accumulator per §5.5 mixed-dtype convention.
	acc := borrowFloat32Buffer(hidden)
	defer releaseFloat32Buffer(acc)

	for bagIndex := 0; bagIndex < bags; bagIndex++ {
		startIdx := int(offsets[bagIndex])
		endIdx := len(indices)

		if bagIndex+1 < bags {
			endIdx = int(offsets[bagIndex+1])
		}

		for dimIndex := range hidden {
			acc[dimIndex] = 0
		}

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(indices[elementIndex])

			for dimIndex := 0; dimIndex < hidden; dimIndex++ {
				acc[dimIndex] += (&table[tokenID*hidden+dimIndex]).Float32()
			}
		}

		outOffset := bagIndex * hidden
		for dimIndex := 0; dimIndex < hidden; dimIndex++ {
			out[outOffset+dimIndex] = dtype.NewBfloat16FromFloat32(acc[dimIndex])
		}
	}

	return nil
}

func runEmbeddingBagFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	table, err := args[0].Float16Native()
	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()
	if err != nil {
		return err
	}

	offsets, err := args[2].Int32Native()
	if err != nil {
		return err
	}

	out, err := args[3].Float16Native()
	if err != nil {
		return err
	}

	tableDims := args[0].Shape().Dims()

	if len(tableDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	hidden := tableDims[1]
	bags := len(offsets)

	if len(out) != bags*hidden {
		return tensor.ErrShapeMismatch
	}

	acc := borrowFloat32Buffer(hidden)
	defer releaseFloat32Buffer(acc)

	for bagIndex := 0; bagIndex < bags; bagIndex++ {
		startIdx := int(offsets[bagIndex])
		endIdx := len(indices)

		if bagIndex+1 < bags {
			endIdx = int(offsets[bagIndex+1])
		}

		for dimIndex := range hidden {
			acc[dimIndex] = 0
		}

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(indices[elementIndex])

			for dimIndex := 0; dimIndex < hidden; dimIndex++ {
				acc[dimIndex] += table[tokenID*hidden+dimIndex].Float32()
			}
		}

		outOffset := bagIndex * hidden
		for dimIndex := 0; dimIndex < hidden; dimIndex++ {
			out[outOffset+dimIndex] = dtype.Fromfloat32(acc[dimIndex])
		}
	}

	return nil
}

func runEmbeddingLookup(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	table, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	tableDims := args[0].Shape().Dims()

	if len(tableDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	vocab := tableDims[0]
	hidden := tableDims[1]

	if len(out) != len(indices)*hidden {
		return tensor.ErrShapeMismatch
	}

	for resultIndex, tokenID := range indices {
		if int(tokenID) < 0 || int(tokenID) >= vocab {
			return tensor.ErrShapeMismatch
		}

		copy(
			out[resultIndex*hidden:(resultIndex+1)*hidden],
			table[int(tokenID)*hidden:(int(tokenID)+1)*hidden],
		)
	}

	return nil
}

/*
EmbeddingBag sums embeddings within each bag. Args: (table,
indices, offsets, output). offsets[b] gives the start of bag b in
indices; bag b spans indices[offsets[b] : offsets[b+1]] or to len
when b is the last bag.
*/
func runEmbeddingBag(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	table, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	indices, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	offsets, err := args[2].Int32Native()

	if err != nil {
		return err
	}

	out, err := args[3].Float32Native()

	if err != nil {
		return err
	}

	tableDims := args[0].Shape().Dims()

	if len(tableDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	hidden := tableDims[1]
	bags := len(offsets)

	if len(out) != bags*hidden {
		return tensor.ErrShapeMismatch
	}

	for bagIndex := 0; bagIndex < bags; bagIndex++ {
		startIdx := int(offsets[bagIndex])

		endIdx := len(indices)

		if bagIndex+1 < bags {
			endIdx = int(offsets[bagIndex+1])
		}

		outOffset := bagIndex * hidden

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(indices[elementIndex])

			for dimIndex := 0; dimIndex < hidden; dimIndex++ {
				out[outOffset+dimIndex] += table[tokenID*hidden+dimIndex]
			}
		}
	}

	return nil
}
