package transport

import (
	"context"
	"testing"

	"capnproto.org/go/capnp/v3"
	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/network/dht"
	"github.com/theapemachine/caramba/pkg/network/schema"
)

func TestNewNodeTransport(t *testing.T) {
	Convey("Given a transport without a DHT", t, func() {
		Convey("When constructing it", func() {
			transport, err := NewNodeTransport("127.0.0.1:0", nil)

			Convey("It should reject the missing DHT", func() {
				So(transport, ShouldBeNil)
				So(err, ShouldNotBeNil)
			})
		})
	})
}

func TestStreamBackendRegistry(t *testing.T) {
	Convey("Given a backend registry with only CPU available", t, func() {
		registry := NewStreamBackendRegistry()

		Convey("When a worker profile advertises CUDA", func() {
			backend, err := registry.NewForProfile(dht.ComputeProfile{
				AvailableRunners: []string{"cuda"},
			})

			Convey("It should not silently fall back to CPU", func() {
				So(backend, ShouldBeNil)
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "cuda")
			})
		})
	})
}

func TestNodeTransport(t *testing.T) {
	Convey("Given two transport instances acting as local and remote nodes", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		localNode, _ := dht.NewNode("127.0.0.1:0", dht.ComputeProfile{})
		localDHT := dht.NewDHT(localNode, nil)
		localTransport, err := NewNodeTransport("127.0.0.1:0", localDHT)
		So(err, ShouldBeNil)
		err = localTransport.Listen()
		So(err, ShouldBeNil)
		defer localTransport.Close()

		remoteNode, _ := dht.NewNode("127.0.0.1:0", dht.ComputeProfile{})
		remoteDHT := dht.NewDHT(remoteNode, nil)
		remoteTransport, err := NewNodeTransport("127.0.0.1:0", remoteDHT)
		So(err, ShouldBeNil)
		err = remoteTransport.Listen()
		So(err, ShouldBeNil)
		defer remoteTransport.Close()

		Convey("When establishing an RPC connection and calling Ping", func() {
			peerClient, rpcConn, err := localTransport.Connect(ctx, remoteTransport.Address())
			So(err, ShouldBeNil)
			defer rpcConn.Close()

			future, release := peerClient.Ping(ctx, func(params schema.Peer_ping_Params) error {
				sender, err := params.NewSender()
				if err != nil {
					return err
				}
				sender.SetId(localNode.ID[:])
				sender.SetAddress(localTransport.Address())
				return nil
			})
			defer release()

			res, err := future.Struct()
			So(err, ShouldBeNil)

			responder, err := res.Responder()
			So(err, ShouldBeNil)

			addr, _ := responder.Address()
			So(addr, ShouldEqual, remoteTransport.Address())

			idData, _ := responder.Id()
			So(idData, ShouldResemble, remoteNode.ID[:])
		})

		Convey("When calling Ping with a malformed sender ID", func() {
			peerClient, rpcConn, err := localTransport.Connect(ctx, remoteTransport.Address())
			So(err, ShouldBeNil)
			defer rpcConn.Close()

			future, release := peerClient.Ping(ctx, func(params schema.Peer_ping_Params) error {
				sender, err := params.NewSender()
				if err != nil {
					return err
				}
				sender.SetId([]byte{1, 2, 3})
				sender.SetAddress(localTransport.Address())
				return nil
			})
			defer release()

			_, err = future.Struct()
			So(err, ShouldNotBeNil)
		})

		Convey("When calling FindNode with a malformed target ID", func() {
			peerClient, rpcConn, err := localTransport.Connect(ctx, remoteTransport.Address())
			So(err, ShouldBeNil)
			defer rpcConn.Close()

			future, release := peerClient.FindNode(ctx, func(params schema.Peer_findNode_Params) error {
				sender, err := params.NewSender()
				if err != nil {
					return err
				}
				sender.SetId(localNode.ID[:])
				sender.SetAddress(localTransport.Address())
				params.SetTargetId([]byte{1, 2, 3})
				return nil
			})
			defer release()

			_, err = future.Struct()
			So(err, ShouldNotBeNil)
		})

		Convey("When calling StartCompute to stream data", func() {
			peerClient, rpcConn, err := localTransport.Connect(ctx, remoteTransport.Address())
			So(err, ShouldBeNil)
			defer rpcConn.Close()

			future, release := peerClient.StartCompute(ctx, func(params schema.Peer_startCompute_Params) error {
				params.SetGraphId("test_graph_123")
				return nil
			})
			defer release()

			stream := future.Stream()

			// Write a node to the stream
			err = stream.WriteNode(ctx, func(params schema.ComputeStream_writeNode_Params) error {
				node, _ := params.NewNode()
				node.SetId("node_1")
				node.SetOp("Input")
				node.SetConfig([]byte(`{"target":true}`))
				return nil
			})
			So(err, ShouldBeNil)

			values, err := encodeFloat64Values([]float64{42})
			So(err, ShouldBeNil)

			err = stream.WriteTensor(ctx, func(params schema.ComputeStream_writeTensor_Params) error {
				tensor, err := params.NewTensor()
				if err != nil {
					return err
				}
				tensor.SetId("node_1")
				shape, err := tensor.NewShape(1)
				if err != nil {
					return err
				}
				shape.Set(0, 1)
				tensor.SetData(values)
				tensor.SetDtype("float64")
				return nil
			})
			So(err, ShouldBeNil)

			// Call execute
			execFuture, execRelease := stream.Execute(ctx, func(params schema.ComputeStream_execute_Params) error {
				return nil
			})
			defer execRelease()

			res, err := execFuture.Struct()
			So(err, ShouldBeNil)

			err = stream.WaitStreaming()
			So(err, ShouldBeNil)

			outputs, err := res.Outputs()
			So(err, ShouldBeNil)
			So(outputs.Len(), ShouldEqual, 1)

			outTensor := outputs.At(0)
			outId, _ := outTensor.Id()
			So(outId, ShouldEqual, "node_1")

			outputData, err := outTensor.Data()
			So(err, ShouldBeNil)
			outputValues, err := decodeFloat64Values(outputData)
			So(err, ShouldBeNil)
			So(outputValues, ShouldResemble, []float64{42})

			workerAddress, err := res.WorkerAddress()
			So(err, ShouldBeNil)
			So(workerAddress, ShouldNotBeBlank)

			workerSignature, err := res.WorkerSignature()
			So(err, ShouldBeNil)
			So(workerSignature, ShouldNotBeEmpty)
		})
	})
}

func TestComputeStreamServerWriteNode(t *testing.T) {
	Convey("Given a Cap'n Proto IR node", t, func() {
		_, segment, err := capnp.NewMessage(capnp.SingleSegment(nil))
		So(err, ShouldBeNil)

		node, err := schema.NewRootIRNode(segment)
		So(err, ShouldBeNil)
		node.SetId("node_1")
		node.SetOp("matmul")
		inputs, err := node.NewInputs(2)
		So(err, ShouldBeNil)
		inputs.Set(0, "left")
		inputs.Set(1, "right")
		node.SetConfig([]byte(`{"transpose":false}`))

		Convey("When copying it into stream-owned memory", func() {
			ownedNode, err := newStreamNode(node)
			So(err, ShouldBeNil)

			node.SetId("mutated")
			inputs.Set(0, "mutated")

			Convey("It should not retain the Cap'n Proto message view", func() {
				So(ownedNode.ID, ShouldEqual, "node_1")
				So(string(ownedNode.Op), ShouldEqual, "Matmul")
				So(ownedNode.Inputs, ShouldResemble, []string{"left", "right"})
				So(ownedNode.Config, ShouldResemble, []byte(`{"transpose":false}`))
			})
		})
	})
}

func TestComputeStreamServerWriteTensor(t *testing.T) {
	Convey("Given a Cap'n Proto tensor", t, func() {
		_, segment, err := capnp.NewMessage(capnp.SingleSegment(nil))
		So(err, ShouldBeNil)

		tensor, err := schema.NewRootTensor(segment)
		So(err, ShouldBeNil)
		tensor.SetId("tensor_1")
		shape, err := tensor.NewShape(2)
		So(err, ShouldBeNil)
		shape.Set(0, 2)
		shape.Set(1, 3)
		tensor.SetData([]byte{1, 2, 3, 4})
		tensor.SetDtype("float32")

		Convey("When copying it into stream-owned memory", func() {
			ownedTensor, err := newStreamTensor(tensor)
			So(err, ShouldBeNil)

			tensor.SetId("mutated")
			shape.Set(0, 99)

			Convey("It should not retain the Cap'n Proto message view", func() {
				So(ownedTensor.ID, ShouldEqual, "tensor_1")
				So(ownedTensor.Shape, ShouldResemble, []int64{2, 3})
				So(ownedTensor.Data, ShouldResemble, []byte{1, 2, 3, 4})
				So(ownedTensor.Dtype, ShouldEqual, "float32")
			})
		})
	})
}
