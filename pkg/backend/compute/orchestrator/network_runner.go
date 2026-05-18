package orchestrator

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	"github.com/theapemachine/caramba/pkg/network/dht"
	"github.com/theapemachine/caramba/pkg/network/schema"
	"github.com/theapemachine/caramba/pkg/network/transport"
	"github.com/theapemachine/caramba/pkg/notary"
)

/*
NetworkRunner implements the Runner interface for executing graphs on remote
volunteer compute nodes across the distributed peer-to-peer network via
Cap'n Proto RPC streaming capabilities.
*/
type NetworkRunner struct {
	notary     *notary.Notary
	identity   *notary.Identity
	dht        *dht.DHT
	transport  *transport.NodeTransport
	targetHW   string // E.g., "cuda", "metal"
	creditCost int64
}

type networkNodeConfig struct {
	Version    int                     `json:"version"`
	Target     bool                    `json:"target"`
	Activation string                  `json:"activation"`
	Operation  string                  `json:"operation"`
	Attributes map[string]ir.Attribute `json:"attributes"`
}

/*
NewNetworkRunner instantiates a remote runner.
*/
func NewNetworkRunner(
	n *notary.Notary,
	id *notary.Identity,
	d *dht.DHT,
	t *transport.NodeTransport,
	targetHardware string,
) *NetworkRunner {
	return &NetworkRunner{
		notary:     n,
		identity:   id,
		dht:        d,
		transport:  t,
		targetHW:   targetHardware,
		creditCost: 100, // Fixed cost for simplicity in this implementation
	}
}

/*
Location returns tensor.Network to indicate remote execution.
*/
func (runner *NetworkRunner) Location() tensor.Location {
	return tensor.Network
}

/*
Close cleans up any network resources if needed (handled by transport).
*/
func (runner *NetworkRunner) Close() error {
	return nil
}

/*
Execute discovers a peer capable of running the required hardware, escrows
the required credits via the Notary ledger, connects to the peer via Cap'n Proto
RPC, streams the graph nodes, triggers execution, and settles the payment upon
successful verification.
*/
func (runner *NetworkRunner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Tensor, error) {
	if graph == nil {
		return nil, fmt.Errorf("network_runner: nil graph")
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("network_runner: empty targets")
	}

	candidates := runner.dht.LookupHardware(ctx, runner.targetHW, 1)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("network_runner: no peers found for hardware %s", runner.targetHW)
	}
	destNode := candidates[0]

	manifestData, err := runner.manifestData(graph, targets)
	if err != nil {
		return nil, err
	}

	signature := runner.identity.Sign(manifestData)
	_, err = runner.notary.SubmitManifest(runner.identity, manifestData, signature, runner.creditCost)
	if err != nil {
		return nil, fmt.Errorf("network_runner: failed to escrow credits: %w", err)
	}

	peerClient, rpcConn, err := runner.transport.Connect(ctx, destNode.Address)
	if err != nil {
		return nil, fmt.Errorf("network_runner: failed to connect to peer: %w", err)
	}
	defer rpcConn.Close()

	computeFuture, computeRelease := peerClient.StartCompute(ctx, func(params schema.Peer_startCompute_Params) error {
		return params.SetGraphId(hex.EncodeToString(runner.identity.Sign(manifestData)))
	})
	defer computeRelease()

	stream := computeFuture.Stream()
	targetIDs := targetSet(targets)

	for _, node := range graph.Nodes() {
		err := stream.WriteNode(ctx, func(params schema.ComputeStream_writeNode_Params) error {
			msgNode, err := params.NewNode()
			if err != nil {
				return err
			}
			if err := msgNode.SetId(node.ID()); err != nil {
				return err
			}
			if err := msgNode.SetOp(string(node.OperationID())); err != nil {
				return err
			}
			if err := setNodeInputs(msgNode, node); err != nil {
				return err
			}
			activation, _ := node.Metadata()["activation"].(string)
			config, err := json.Marshal(networkNodeConfig{
				Version:    1,
				Target:     targetIDs[node.ID()],
				Activation: activation,
				Operation:  string(node.OperationID()),
				Attributes: node.Attributes(),
			})
			if err != nil {
				return err
			}
			if err := msgNode.SetConfig(config); err != nil {
				return err
			}

			return nil
		})

		if err != nil {
			return nil, fmt.Errorf("network_runner: failed to stream node %s: %w", node.ID(), err)
		}

		if err := runner.streamInputTensor(ctx, stream, node); err != nil {
			return nil, err
		}
	}

	execFuture, execRelease := stream.Execute(ctx, func(params schema.ComputeStream_execute_Params) error {
		return nil
	})
	defer execRelease()

	res, err := execFuture.Struct()
	if err != nil {
		return nil, fmt.Errorf("network_runner: remote execution failed: %w", err)
	}

	if err := stream.WaitStreaming(); err != nil {
		return nil, fmt.Errorf("network_runner: streaming failed: %w", err)
	}

	resError, _ := res.Error()
	if resError != "" {
		return nil, fmt.Errorf("network_runner: remote node returned error: %s", resError)
	}

	outputs, err := res.Outputs()
	if err != nil {
		return nil, err
	}

	resultData, err := transport.ResultProofPayloadFromSchema(outputs)
	if err != nil {
		return nil, err
	}

	workerIdentity, err := workerIdentityFromResults(res)
	if err != nil {
		return nil, err
	}

	workerSignature, err := res.WorkerSignature()
	if err != nil {
		return nil, err
	}

	err = runner.notary.SettleCompute(
		runner.identity.Address(),
		workerIdentity,
		resultData,
		workerSignature,
		runner.creditCost,
	)
	if err != nil {
		return nil, fmt.Errorf("network_runner: failed to settle payment: %w", err)
	}

	return tensorsFromResults(outputs)
}

func (runner *NetworkRunner) streamInputTensor(
	ctx context.Context, stream schema.ComputeStream, node *ir.Node,
) error {
	if node.OpType() != ir.OpInput {
		return nil
	}

	rawValues, ok := node.Metadata()["values"]
	if !ok {
		return fmt.Errorf("network_runner: input node %s has no values metadata", node.ID())
	}

	values, ok := rawValues.([]float64)
	if !ok {
		return fmt.Errorf("network_runner: input node %s values metadata must be []float64", node.ID())
	}

	data, err := encodeFloat64Values(values)
	if err != nil {
		return err
	}

	err = stream.WriteTensor(ctx, func(params schema.ComputeStream_writeTensor_Params) error {
		msgTensor, err := params.NewTensor()
		if err != nil {
			return err
		}
		if err := msgTensor.SetId(node.ID()); err != nil {
			return err
		}
		if err := setTensorShape(msgTensor, node.Shape()); err != nil {
			return err
		}
		if err := msgTensor.SetData(data); err != nil {
			return err
		}

		return msgTensor.SetDtype(dtype.Float64.String())
	})

	if err != nil {
		return fmt.Errorf("network_runner: failed to stream tensor %s: %w", node.ID(), err)
	}

	return nil
}

func (runner *NetworkRunner) manifestData(graph *ir.Graph, targets []*ir.Node) ([]byte, error) {
	buffer := bytes.NewBuffer(nil)
	if err := writeProofString(buffer, runner.targetHW); err != nil {
		return nil, err
	}

	for _, node := range graph.Nodes() {
		if err := writeProofString(buffer, node.ID()); err != nil {
			return nil, err
		}
		if err := writeProofString(buffer, string(node.OpType())); err != nil {
			return nil, err
		}
	}

	for _, target := range targets {
		if err := writeProofString(buffer, target.ID()); err != nil {
			return nil, err
		}
	}

	return buffer.Bytes(), nil
}

func setNodeInputs(msgNode schema.IRNode, node *ir.Node) error {
	inputs := node.Inputs()
	inputList, err := msgNode.NewInputs(int32(len(inputs)))
	if err != nil {
		return err
	}

	for index, input := range inputs {
		if err := inputList.Set(index, input.ID()); err != nil {
			return err
		}
	}

	return nil
}

func setTensorShape(msgTensor schema.Tensor, shape tensor.Shape) error {
	dimensions := shape.Dims()
	shapeList, err := msgTensor.NewShape(int32(len(dimensions)))
	if err != nil {
		return err
	}

	for index, dimension := range dimensions {
		shapeList.Set(index, int64(dimension))
	}

	return nil
}

func targetSet(targets []*ir.Node) map[string]bool {
	set := make(map[string]bool, len(targets))

	for _, target := range targets {
		set[target.ID()] = true
	}

	return set
}

func workerIdentityFromResults(res schema.ComputeStream_execute_Results) (*notary.Identity, error) {
	workerAddress, err := res.WorkerAddress()
	if err != nil {
		return nil, err
	}

	publicKey, err := hex.DecodeString(workerAddress)
	if err != nil {
		return nil, err
	}

	if len(publicKey) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("network_runner: invalid worker address")
	}

	return &notary.Identity{PublicKey: ed25519.PublicKey(publicKey)}, nil
}

func tensorsFromResults(outputs schema.Tensor_List) (map[string]tensor.Tensor, error) {
	hostBackend := tensor.NewHostBackend()
	results := make(map[string]tensor.Tensor, outputs.Len())

	for index := 0; index < outputs.Len(); index++ {
		output := outputs.At(index)
		id, err := output.Id()
		if err != nil {
			return nil, err
		}
		rawDType, err := output.Dtype()
		if err != nil {
			return nil, err
		}
		sourceDType, err := dtype.Parse(rawDType)
		if err != nil {
			return nil, err
		}

		shapeList, err := output.Shape()
		if err != nil {
			return nil, err
		}
		dimensions := make([]int, shapeList.Len())
		for dimensionIndex := range dimensions {
			dimensions[dimensionIndex] = int(shapeList.At(dimensionIndex))
		}
		shape, err := tensor.NewShape(dimensions)
		if err != nil {
			return nil, err
		}
		data, err := output.Data()
		if err != nil {
			return nil, err
		}
		uploaded, err := hostBackend.Upload(shape, sourceDType, data)
		if err != nil {
			return nil, err
		}

		results[id] = uploaded
	}

	return results, nil
}

func encodeFloat64Values(values []float64) ([]byte, error) {
	buffer := bytes.NewBuffer(make([]byte, 0, len(values)*8))

	for _, value := range values {
		if err := binary.Write(buffer, binary.LittleEndian, value); err != nil {
			return nil, err
		}
	}

	return buffer.Bytes(), nil
}

func decodeFloat64Values(data []byte) ([]float64, error) {
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("network_runner: float64 tensor data must be divisible by 8")
	}

	values := make([]float64, len(data)/8)
	reader := bytes.NewReader(data)

	for index := range values {
		if err := binary.Read(reader, binary.LittleEndian, &values[index]); err != nil {
			return nil, err
		}
	}

	return values, nil
}

func writeProofString(buffer *bytes.Buffer, value string) error {
	if err := binary.Write(buffer, binary.LittleEndian, uint64(len(value))); err != nil {
		return err
	}

	_, err := buffer.WriteString(value)

	return err
}
