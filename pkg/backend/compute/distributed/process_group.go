package distributed

import (
	"context"
	"sync"

	"github.com/theapemachine/manifesto/tensor"
)

/*
ProcessGroup is the bootstrap surface for multi-process distributed
runs. The host-only test path uses LocalProcessGroup which keeps
every rank in the same address space. Real deployments register a
network-backed implementation (pkg/network/transport will host the
NCCL-style or gRPC rendezvous variant in a follow-up session).

Per the spray-and-pray contract, this file establishes the
ProcessGroup interface and the local-only reference. The
rendezvous protocol, peer discovery, and failure detection are
defined here but only LocalProcessGroup implements them; the
network variant lands when there's a Linux-with-NCCL host to
verify against.
*/
type ProcessGroup interface {
	// Rank returns the calling process's rank within the group.
	Rank() int

	// Size returns the number of processes in the group.
	Size() int

	// Barrier blocks until every process in the group has called
	// Barrier with the same tag. The tag is a sequence number used
	// to match calls across processes.
	Barrier(ctx context.Context, tag uint64) error

	// AllReduce performs an in-place collective AllReduce across
	// the group. Each rank contributes its shard; on return every
	// rank's tensor holds the reduced value.
	AllReduce(ctx context.Context, op CollectiveOp, target tensor.Tensor) error

	// Close releases the group's resources.
	Close() error
}

/*
CollectiveOp identifies the reduction in cross-process AllReduce.
Mirrors the in-process collective.Op enum so callers don't need to
hold both types.
*/
type CollectiveOp uint8

const (
	OpSum CollectiveOp = iota
	OpMax
	OpMin
	OpMean
)

/*
LocalProcessGroup is the in-memory ProcessGroup used by single-
process tests and by the host-only distributed reference. Every
"rank" is a goroutine sharing the same memory; Barrier and AllReduce
synchronize through sync.Cond.
*/
type LocalProcessGroup struct {
	rank  int
	size  int
	state *localGroupState
}

type localGroupState struct {
	mu            sync.Mutex
	cond          *sync.Cond
	barrierTag    uint64
	barrierHit    int
	allReduceIn   []tensor.Tensor
	allReduceDone bool
	allReduceLeft int
	closed        bool
}

/*
NewLocalProcessGroup constructs a slice of process-group handles
sharing the same in-memory state. Each handle represents one rank;
the returned slice has length size. Callers spawn one goroutine per
handle and run the distributed logic in each.
*/
func NewLocalProcessGroup(size int) []*LocalProcessGroup {
	state := &localGroupState{}
	state.cond = sync.NewCond(&state.mu)

	groups := make([]*LocalProcessGroup, size)

	for rankIndex := range groups {
		groups[rankIndex] = &LocalProcessGroup{
			rank:  rankIndex,
			size:  size,
			state: state,
		}
	}

	return groups
}

/*
Rank returns the calling handle's rank.
*/
func (group *LocalProcessGroup) Rank() int { return group.rank }

/*
Size returns the group size.
*/
func (group *LocalProcessGroup) Size() int { return group.size }

/*
Barrier waits until every rank in the group has hit the same tag.
*/
func (group *LocalProcessGroup) Barrier(ctx context.Context, tag uint64) error {
	group.state.mu.Lock()
	defer group.state.mu.Unlock()

	if group.state.closed {
		return tensor.ErrBackendClosed
	}

	if group.state.barrierTag != tag {
		group.state.barrierTag = tag
		group.state.barrierHit = 0
	}

	group.state.barrierHit++

	if group.state.barrierHit == group.size {
		group.state.cond.Broadcast()
		return nil
	}

	for group.state.barrierHit < group.size && !group.state.closed {
		group.state.cond.Wait()
	}

	if group.state.closed {
		return tensor.ErrBackendClosed
	}

	return nil
}

/*
AllReduce performs a per-rank in-place reduction. The reference
implementation in this file is a single-process synchronization: each
rank contributes its tensor to the shared input slice; the rank that
sees the last contribution applies the reduction once, sets
allReduceDone, and broadcasts to wake the other ranks.
*/
func (group *LocalProcessGroup) AllReduce(
	ctx context.Context,
	op CollectiveOp,
	target tensor.Tensor,
) error {
	group.state.mu.Lock()

	if group.state.allReduceIn == nil {
		group.state.allReduceIn = make([]tensor.Tensor, group.size)
		group.state.allReduceLeft = group.size
		group.state.allReduceDone = false
	}

	group.state.allReduceIn[group.rank] = target
	group.state.allReduceLeft--

	if group.state.allReduceLeft == 0 {
		shards := group.state.allReduceIn

		_ = applyLocalAllReduce(op, shards)

		group.state.allReduceDone = true
		group.state.cond.Broadcast()
	}

	for !group.state.allReduceDone && !group.state.closed {
		group.state.cond.Wait()
	}

	closed := group.state.closed

	// The last rank to leave resets the shared state for the next
	// AllReduce call.
	if group.state.allReduceDone {
		group.state.allReduceIn[group.rank] = nil
		stillPresent := false

		for _, candidate := range group.state.allReduceIn {
			if candidate != nil {
				stillPresent = true
				break
			}
		}

		if !stillPresent {
			group.state.allReduceIn = nil
			group.state.allReduceDone = false
		}
	}

	group.state.mu.Unlock()

	if closed {
		return tensor.ErrBackendClosed
	}

	return nil
}

/*
Close marks the group closed. Outstanding Barrier and AllReduce
callers wake up and return ErrBackendClosed.
*/
func (group *LocalProcessGroup) Close() error {
	group.state.mu.Lock()
	defer group.state.mu.Unlock()

	group.state.closed = true
	group.state.cond.Broadcast()
	return nil
}

func applyLocalAllReduce(op CollectiveOp, shards []tensor.Tensor) error {
	if len(shards) == 0 {
		return nil
	}

	views := make([][]float32, len(shards))

	for index, shard := range shards {
		view, err := shard.Float32Native()

		if err != nil {
			return err
		}

		views[index] = view
	}

	length := len(views[0])
	accumulator := make([]float32, length)

	switch op {
	case OpSum, OpMean:
		for _, view := range views {
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
			for index, value := range view {
				if value > accumulator[index] {
					accumulator[index] = value
				}
			}
		}
	case OpMin:
		copy(accumulator, views[0])

		for _, view := range views[1:] {
			for index, value := range view {
				if value < accumulator[index] {
					accumulator[index] = value
				}
			}
		}
	}

	for _, view := range views {
		copy(view, accumulator)
	}

	return nil
}

var _ ProcessGroup = (*LocalProcessGroup)(nil)
