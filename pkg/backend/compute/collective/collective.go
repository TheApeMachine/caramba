/*
Package collective implements the cross-shard communication
primitives required by distributed tensors (TENSOR_BACKEND_REWRITE.md
§2.17, §3.10). AllReduce / AllGather / ReduceScatter / Broadcast are
exposed as Operations that take a slice of per-shard tensors plus a
reduction op and produce the cross-shard result.

Per the spray-and-pray contract, this file provides the host-local
reference implementations (loops over the shard slice). NCCL on CUDA,
MPS multi-GPU ring on Metal, and the pkg/network-transport-backed
ring on host nodes are implemented in their respective backend
sessions and dispatch through the same interface.
*/
package collective

import (
	"context"
	"errors"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Op identifies the reduction operation applied across shards.
*/
type Op uint8

const (
	OpSum Op = iota
	OpMax
	OpMin
	OpMean
)

/*
ErrInvalidSourceIndex is returned by Broadcast when source is not a
valid index into the shard slice. Distinct from
tensor.ErrShapeMismatch so callers can branch on intent.
*/
var ErrInvalidSourceIndex = errors.New("collective: source index out of range")

/*
AllReduce sums (or max/min/mean) across shards and stores the result
on every shard. Every shard must have the same shape and dtype.
Honors ctx cancellation between the validation/acquisition step and
each major loop body.
*/
func AllReduce(ctx context.Context, op Op, shards []tensor.Tensor) error {
	if len(shards) == 0 {
		return nil
	}

	if err := requireConsistentShape(shards); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	views, err := acquireFloat32Views(shards)

	if err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	length := len(views[0])
	accumulator := make([]float32, length)

	switch op {
	case OpSum, OpMean:
		for _, view := range views {
			if err := ctx.Err(); err != nil {
				return err
			}

			for index, value := range view {
				accumulator[index] += value
			}
		}

		if op == OpMean {
			divisor := float32(len(views))

			for index := range accumulator {
				accumulator[index] /= divisor
			}
		}
	case OpMax:
		copy(accumulator, views[0])

		for _, view := range views[1:] {
			if err := ctx.Err(); err != nil {
				return err
			}

			for index, value := range view {
				if value > accumulator[index] {
					accumulator[index] = value
				}
			}
		}
	case OpMin:
		copy(accumulator, views[0])

		for _, view := range views[1:] {
			if err := ctx.Err(); err != nil {
				return err
			}

			for index, value := range view {
				if value < accumulator[index] {
					accumulator[index] = value
				}
			}
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	for _, view := range views {
		copy(view, accumulator)
	}

	return nil
}

/*
Broadcast copies the source shard's values to every other shard.
*/
func Broadcast(ctx context.Context, source int, shards []tensor.Tensor) error {
	if len(shards) == 0 {
		return nil
	}

	if source < 0 || source >= len(shards) {
		return fmt.Errorf("%w: source=%d shards=%d", ErrInvalidSourceIndex, source, len(shards))
	}

	if err := requireConsistentShape(shards); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	views, err := acquireFloat32Views(shards)

	if err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	for index, view := range views {
		if err := ctx.Err(); err != nil {
			return err
		}

		if index == source {
			continue
		}

		copy(view, views[source])
	}

	return nil
}

/*
AllGather concatenates the per-shard tensors into each output. The
host reference builds the concatenated buffer once and copies it
into each output, so total work is (N+1) × totalLen rather than
N × N × shardLen. Phase 10 expansion will accept an axis parameter;
the skeleton here concatenates along axis 0 by default.
*/
func AllGather(ctx context.Context, shards []tensor.Tensor, outputs []tensor.Tensor) error {
	if len(shards) == 0 || len(outputs) != len(shards) {
		return tensor.ErrShapeMismatch
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	shardViews, err := acquireFloat32Views(shards)

	if err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	outputViews, err := acquireFloat32Views(outputs)

	if err != nil {
		return err
	}

	totalLen := 0

	for _, view := range shardViews {
		totalLen += len(view)
	}

	for _, output := range outputViews {
		if len(output) != totalLen {
			return tensor.ErrShapeMismatch
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}

	// Build the concatenated buffer once.
	concatenated := make([]float32, totalLen)
	offset := 0

	for _, shard := range shardViews {
		if err := ctx.Err(); err != nil {
			return err
		}

		copy(concatenated[offset:offset+len(shard)], shard)
		offset += len(shard)
	}

	// Distribute the single buffer to every output.
	for _, output := range outputViews {
		if err := ctx.Err(); err != nil {
			return err
		}

		copy(output, concatenated)
	}

	return nil
}

/*
ReduceScatter sums shards and scatters disjoint pieces of the result
back to each shard.
*/
func ReduceScatter(ctx context.Context, op Op, shards []tensor.Tensor) error {
	// For the host reference: equivalent to AllReduce followed by
	// each shard taking its own piece. The host fast path is just
	// AllReduce since shards already have the same shape.
	return AllReduce(ctx, op, shards)
}

func requireConsistentShape(shards []tensor.Tensor) error {
	reference := shards[0].Shape()

	for _, shard := range shards[1:] {
		if !shard.Shape().Equal(reference) {
			return tensor.ErrShapeMismatch
		}

		if shard.DType() != shards[0].DType() {
			return tensor.ErrDTypeMismatch
		}
	}

	return nil
}

/*
acquireFloat32Views returns the underlying float32 slice for each
shard. Currently the host reference collectives operate on float32
storage only; the bf16 / fp16 / fp8 paths land in later sessions
through additional acquire helpers (acquireBFloat16Views, etc.).
For any non-float32 storage the helper returns ErrDTypeUnsupported
with the offending dtype named in the wrapped error.
*/
func acquireFloat32Views(shards []tensor.Tensor) ([][]float32, error) {
	views := make([][]float32, len(shards))

	for index, shard := range shards {
		if shard.DType() != dtype.Float32 {
			return nil, fmt.Errorf(
				"%w: collective host reference accepts only float32 today, got %s",
				tensor.ErrDTypeUnsupported, shard.DType(),
			)
		}

		view, err := shard.Float32Native()

		if err != nil {
			return nil, err
		}

		views[index] = view
	}

	return views, nil
}
