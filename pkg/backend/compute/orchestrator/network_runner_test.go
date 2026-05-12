package orchestrator

import (
	"context"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
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

func (backend *fakeCUDAStreamBackend) UploadFloat64(
	shape tensor.Shape, values []float64,
) (tensor.Float64Tensor, error) {
	return backend.backend.UploadFloat64(shape, values)
}

func (backend *fakeCUDAStreamBackend) DownloadFloat64(
	input tensor.Float64Tensor,
) ([]float64, error) {
	return backend.backend.DownloadFloat64(input)
}

func (backend *fakeCUDAStreamBackend) Close() error {
	return backend.backend.Close()
}

func (backend *fakeCUDAStreamBackend) ReLU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.ReLU(input)
}

func (backend *fakeCUDAStreamBackend) LeakyReLU(
	input tensor.Float64Tensor, alpha float64,
) (tensor.Float64Tensor, error) {
	return backend.backend.LeakyReLU(input, alpha)
}

func (backend *fakeCUDAStreamBackend) GELU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.GELU(input)
}

func (backend *fakeCUDAStreamBackend) Tanh(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.Tanh(input)
}

func (backend *fakeCUDAStreamBackend) Sigmoid(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.Sigmoid(input)
}

func (backend *fakeCUDAStreamBackend) SwiGLU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.SwiGLU(input)
}

func (backend *fakeCUDAStreamBackend) Add(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.Add(left, right)
}

func (backend *fakeCUDAStreamBackend) Mul(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.Mul(left, right)
}

func (backend *fakeCUDAStreamBackend) Matmul(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.Matmul(left, right)
}

func (backend *fakeCUDAStreamBackend) MatmulAdd(
	left, right, bias tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.MatmulAdd(left, right, bias)
}

func (backend *fakeCUDAStreamBackend) MatmulAddGELU(
	left, right, bias tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return backend.backend.MatmulAddGELU(left, right, bias)
}

func TestNetworkRunner(t *testing.T) {
	Convey("Given a NetworkRunner and a functioning network ecosystem", t, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// 1. Setup Notary
		ntry := notary.NewNotary()
		localId, err := notary.NewIdentity()
		So(err, ShouldBeNil)
		ntry.Ledger().Mint(localId.Address(), 1000)
		ntry.Ledger().Mint("escrow", 0)

		// 2. Setup DHT for Remote
		remoteNode, err := dht.NewNode("127.0.0.1:0", dht.ComputeProfile{
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
		localNode, err := dht.NewNode("127.0.0.1:0", dht.ComputeProfile{})
		So(err, ShouldBeNil)
		localDHT := dht.NewDHT(localNode, nil)

		// Register the actual bound address of the remote node
		actualRemoteAddr := remoteTransport.Address()
		knownRemoteNode, err := dht.NewNode(actualRemoteAddr, dht.ComputeProfile{
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

			values, err := results["test"].CloneFloat64()
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
