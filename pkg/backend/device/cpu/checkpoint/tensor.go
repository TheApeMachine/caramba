package checkpoint

import (
	"encoding/binary"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func RunCheckpointEncodeFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Uint8Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()
	headerBytes := 16 + len(dims)*8
	dataBytes := len(input) * 4

	if len(out) != headerBytes+dataBytes {
		return tensor.ErrShapeMismatch
	}

	binary.LittleEndian.PutUint64(out[0:8], uint64(len(dims)))
	binary.LittleEndian.PutUint64(out[8:16], uint64(dataBytes))

	for index, dim := range dims {
		binary.LittleEndian.PutUint64(out[16+index*8:], uint64(dim))
	}

	dataOffset := headerBytes

	for index, value := range input {
		binary.LittleEndian.PutUint32(out[dataOffset+index*4:], math.Float32bits(value))
	}

	return nil
}

func RunCheckpointDecodeFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	in, err := args[0].Uint8Native()

	if err != nil {
		return err
	}

	out, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	if len(in) < 16 {
		return tensor.ErrShapeMismatch
	}

	rank := int(binary.LittleEndian.Uint64(in[0:8]))
	dataBytes := int(binary.LittleEndian.Uint64(in[8:16]))
	headerBytes := 16 + rank*8

	if len(in) != headerBytes+dataBytes || len(out)*4 != dataBytes {
		return tensor.ErrShapeMismatch
	}

	for index := range out {
		out[index] = math.Float32frombits(binary.LittleEndian.Uint32(in[headerBytes+index*4:]))
	}

	return nil
}
