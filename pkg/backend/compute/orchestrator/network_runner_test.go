package orchestrator

import (
	"context"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	"github.com/theapemachine/caramba/pkg/network/dht"
	"github.com/theapemachine/caramba/pkg/network/transport"
	"github.com/theapemachine/caramba/pkg/notary"
)

type fakeCUDAStreamBackend struct {
	backend *computecpu.TensorBackend
}

func newFakeCUDAStreamBackend() (transport.StreamComputeBackend, error) {
	return &fakeCUDAStreamBackend{backend: computecpu.NewTensorBackend()}, nil
}

func (backend *fakeCUDAStreamBackend) Location() tensor.Location {
	return tensor.CUDA
}

func (backend *fakeCUDAStreamBackend) SupportedDTypes() []dtype.DType {
	return backend.backend.SupportedDTypes()
}

func (backend *fakeCUDAStreamBackend) SupportedLayouts() []tensor.Layout {
	return backend.backend.SupportedLayouts()
}

func (backend *fakeCUDAStreamBackend) Capabilities() tensor.Capabilities {
	return backend.backend.Capabilities()
}

func (backend *fakeCUDAStreamBackend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (tensor.Tensor, error) {
	return backend.backend.Upload(shape, sourceDType, bytes)
}

func (backend *fakeCUDAStreamBackend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	bytes []byte,
) (tensor.Tensor, error) {
	return backend.backend.UploadAsync(shape, sourceDType, bytes)
}

func (backend *fakeCUDAStreamBackend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return backend.backend.UploadSparse(shape, valueDType, layout, values, indices)
}

func (backend *fakeCUDAStreamBackend) Download(
	input tensor.Tensor,
) (dtype.DType, []byte, error) {
	return backend.backend.Download(input)
}

func (backend *fakeCUDAStreamBackend) Close() error {
	return backend.backend.Close()
}

func (backend *fakeCUDAStreamBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Tensor,
) (tensor.Tensor, error) {
	return backend.backend.Apply(ctx, node, inputs)
}

func TestNetworkRunner(t *testing.T) {
	Convey("Given a NetworkRunner and a functioning network ecosystem", t, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// 1. Setup Notary
		ntry := notary.NewNotary()
		localId, err := notary.NewIdentity()
		So(err, ShouldBeNil)
		err = ntry.MintCredits(localId.Address(), 1000)
		So(err, ShouldBeNil)

		// 2. Setup DHT for Remote
		remoteIdentity, err := notary.NewIdentity()
		So(err, ShouldBeNil)
		remoteNode, err := dht.NewNode("127.0.0.1:0", remoteIdentity.Address(), dht.ComputeProfile{
			AvailableRunners: []string{"cuda"},
		})
		So(err, ShouldBeNil)
		remoteDHT := dht.NewDHT(remoteNode, nil)

		// Setup Remote Transport (PeerServer is automatically attached)
		remoteTransport, err := transport.NewNodeTransport("127.0.0.1:0", remoteDHT)
		So(err, ShouldBeNil)
		remoteTransport.RegisterStreamBackend("cuda", newFakeCUDAStreamBackend)
		err = remoteTransport.Listen()
		So(err, ShouldBeNil)
		defer remoteTransport.Close()

		// 3. Setup DHT for Local
		localNode, err := dht.NewNode("127.0.0.1:0", localId.Address(), dht.ComputeProfile{})
		So(err, ShouldBeNil)
		localDHT := dht.NewDHT(localNode, nil)

		// Register the actual bound address of the remote node
		actualRemoteAddr := remoteTransport.Address()
		knownRemoteNode, err := dht.NewNode(actualRemoteAddr, remoteIdentity.Address(), dht.ComputeProfile{
			AvailableRunners: []string{"cuda"},
		})
		So(err, ShouldBeNil)
		localDHT.AddNode(knownRemoteNode)

		// Setup Local Transport
		localTransport, err := transport.NewNodeTransport("127.0.0.1:0", localDHT)
		So(err, ShouldBeNil)
		err = localTransport.Listen()
		So(err, ShouldBeNil)
		defer localTransport.Close()

		// 4. Create the NetworkRunner
		runner := NewNetworkRunner(ntry, localId, localDHT, localTransport, "cuda")

		Convey("Location should be Network", func() {
			So(runner.Location(), ShouldEqual, tensor.Network)
		})

		Convey("It should fail when graph is nil", func() {
			_, err := runner.Execute(ctx, nil, nil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "nil graph")
		})

		Convey("It should execute successfully via RPC stream capabilities", func() {
			graph := ir.NewGraph()
			shape, _ := tensor.NewShape([]int{1})
			node := ir.NewNode("test", ir.OpInput, shape)
			node.SetMetadata("values", []float64{42})
			graph.AddNode(node)

			results, err := runner.Execute(ctx, graph, []*ir.Node{node})
			So(err, ShouldBeNil)
			So(results, ShouldNotBeNil)
			So(len(results), ShouldEqual, 1)

			values, err := tensorFloat64Values(results["test"])
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{42})
			So(ntry.Ledger().BalanceOf(localId.Address()), ShouldEqual, 900)
			So(ntry.Ledger().BalanceOf("escrow"), ShouldEqual, 0)
		})

		Convey("It should fail when no peers match the hardware requirement", func() {
			emptyDHT := dht.NewDHT(localNode, nil)
			runnerEmpty := NewNetworkRunner(ntry, localId, emptyDHT, localTransport, "tpu")

			graph := ir.NewGraph()
			shape, _ := tensor.NewShape([]int{1})
			node := ir.NewNode("test", "test", shape)
			graph.AddNode(node)

			_, err := runnerEmpty.Execute(ctx, graph, []*ir.Node{node})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no peers found")
		})
	})
}
