/*
Package distributed implements the DistributedTensor abstraction from
TENSOR_BACKEND_REWRITE.md §2.17. A DistributedTensor composes per-shard
Tensor values under a ShardingMesh + ShardingSpec; collective operations
live in pkg/backend/compute/collective.

Per the spray-and-pray contract, this package provides the concrete
HostDistributedTensor implementation (which fits a single process's
host backend across a "mesh of one" — useful for testing collective
patterns) plus the Mesh / Spec utility helpers. Real multi-process
deployments lift the same Tensor handles to a remote-shard wrapper
via pkg/network/transport.
*/
package distributed

import (
	"errors"
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
ErrInvalidRank is returned when a rank index is outside the range
[0, mesh size). Distinct from tensor.ErrShapeMismatch so callers can
distinguish topology errors from tensor-shape errors.
*/
var ErrInvalidRank = errors.New("distributed: rank out of range")

/*
HostDistributedTensor is the host-only DistributedTensor. Every shard
lives in the same process; collective operations are local
gather/scatter loops. Used as the reference implementation against
which multi-process collective backends (NCCL, MPS multi-GPU,
network ring) are tested.
*/
type HostDistributedTensor struct {
	logicalShape tensor.Shape
	storedDType  dtype.DType
	storedLayout tensor.Layout
	mesh         tensor.ShardingMesh
	spec         tensor.ShardingSpec
	shards       []tensor.Tensor
	localRank    int
	closed       atomic.Bool
}

/*
NewHostDistributedTensor builds a host-only DistributedTensor from a
slice of per-mesh-device tensors. The shards slice is taken
wholesale; ownership transfers to the DistributedTensor.
*/
func NewHostDistributedTensor(
	logicalShape tensor.Shape,
	storedDType dtype.DType,
	storedLayout tensor.Layout,
	mesh tensor.ShardingMesh,
	spec tensor.ShardingSpec,
	shards []tensor.Tensor,
	localRank int,
) *HostDistributedTensor {
	return &HostDistributedTensor{
		logicalShape: logicalShape,
		storedDType:  storedDType,
		storedLayout: storedLayout,
		mesh:         mesh,
		spec:         spec,
		shards:       shards,
		localRank:    localRank,
	}
}

/*
LogicalShape returns the unsharded shape.
*/
func (distributed *HostDistributedTensor) LogicalShape() tensor.Shape {
	return distributed.logicalShape
}

/*
DType returns the stored dtype.
*/
func (distributed *HostDistributedTensor) DType() dtype.DType {
	return distributed.storedDType
}

/*
Layout returns the storage layout.
*/
func (distributed *HostDistributedTensor) Layout() tensor.Layout {
	return distributed.storedLayout
}

/*
Mesh returns the device mesh.
*/
func (distributed *HostDistributedTensor) Mesh() tensor.ShardingMesh {
	return distributed.mesh
}

/*
Sharding returns the sharding spec.
*/
func (distributed *HostDistributedTensor) Sharding() tensor.ShardingSpec {
	return distributed.spec
}

/*
Shards returns the per-mesh-device tensors.
*/
func (distributed *HostDistributedTensor) Shards() []tensor.Tensor {
	return distributed.shards
}

/*
LocalShard returns the shard for the local process's rank.
*/
func (distributed *HostDistributedTensor) LocalShard() (tensor.Tensor, error) {
	if distributed.localRank < 0 || distributed.localRank >= len(distributed.shards) {
		return nil, ErrInvalidRank
	}

	return distributed.shards[distributed.localRank], nil
}

/*
Close releases every shard.
*/
func (distributed *HostDistributedTensor) Close() error {
	if !distributed.closed.CompareAndSwap(false, true) {
		return nil
	}

	var firstErr error

	for _, shard := range distributed.shards {
		if err := shard.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

var _ tensor.DistributedTensor = (*HostDistributedTensor)(nil)

/*
MeshRank returns the flat rank for the given mesh coordinates. coords
must have length equal to mesh.Shape's length. The walk is C-order
(last axis fastest).
*/
func MeshRank(mesh tensor.ShardingMesh, coords []int) (int, error) {
	if len(coords) != len(mesh.Shape) {
		return 0, tensor.ErrShapeMismatch
	}

	rank := 0
	stride := 1

	for axisIndex := len(mesh.Shape) - 1; axisIndex >= 0; axisIndex-- {
		if coords[axisIndex] < 0 || coords[axisIndex] >= mesh.Shape[axisIndex] {
			return 0, tensor.ErrShapeMismatch
		}

		rank += coords[axisIndex] * stride
		stride *= mesh.Shape[axisIndex]
	}

	return rank, nil
}

/*
MeshCoords inverts MeshRank.
*/
func MeshCoords(mesh tensor.ShardingMesh, rank int) ([]int, error) {
	total := 1

	for _, size := range mesh.Shape {
		total *= size
	}

	if rank < 0 || rank >= total {
		return nil, tensor.ErrShapeMismatch
	}

	coords := make([]int, len(mesh.Shape))

	for axisIndex := len(mesh.Shape) - 1; axisIndex >= 0; axisIndex-- {
		coords[axisIndex] = rank % mesh.Shape[axisIndex]
		rank /= mesh.Shape[axisIndex]
	}

	return coords, nil
}
