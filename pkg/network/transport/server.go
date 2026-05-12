package transport

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/network/dht"
	"github.com/theapemachine/caramba/pkg/network/schema"
	"github.com/theapemachine/caramba/pkg/notary"
)

const nodeIDLength = 20

type StreamComputeBackend = executor.Backend

type StreamBackendFactory func() (StreamComputeBackend, error)

type StreamBackendRegistry struct {
	factories map[string]StreamBackendFactory
}

func NewStreamBackendRegistry() *StreamBackendRegistry {
	registry := &StreamBackendRegistry{
		factories: make(map[string]StreamBackendFactory),
	}

	registry.Register("cpu", NewCPUStreamBackend)
	registry.Register("host", NewCPUStreamBackend)
	for _, registeredFactory := range acceleratorStreamBackendFactories {
		registry.Register(registeredFactory.name, registeredFactory.factory)
	}

	return registry
}

func (registry *StreamBackendRegistry) Register(name string, factory StreamBackendFactory) {
	if registry == nil || factory == nil {
		return
	}

	registry.factories[normalizeRunnerName(name)] = factory
}

func (registry *StreamBackendRegistry) NewForProfile(
	profile dht.ComputeProfile,
) (StreamComputeBackend, error) {
	if registry == nil {
		registry = NewStreamBackendRegistry()
	}

	runners := profile.AvailableRunners
	if len(runners) == 0 {
		runners = []string{"cpu"}
	}

	runner := normalizeRunnerName(runners[0])
	factory, ok := registry.factories[runner]

	if !ok {
		return nil, fmt.Errorf("transport: no stream compute backend registered for %q", runner)
	}

	backend, err := factory()

	if err != nil {
		return nil, fmt.Errorf("transport: failed to initialize %s backend: %w", runner, err)
	}

	if normalizeRunnerName(string(backend.Location())) != runner &&
		!(runner == "cpu" && backend.Location() == computetensor.Host) {
		_ = backend.Close()

		return nil, fmt.Errorf(
			"transport: backend %q initialized as %q",
			runner,
			backend.Location(),
		)
	}

	return backend, nil
}

func normalizeRunnerName(name string) string {
	switch strings.ToLower(name) {
	case "cpu", "host":
		return "cpu"
	case "cuda":
		return "cuda"
	case "metal":
		return "metal"
	case "xla":
		return "xla"
	default:
		return name
	}
}

/*
PeerServer implements the schema.Peer_Server interface.
It handles Kademlia DHT messages and compute stream requests.
*/
type PeerServer struct {
	DHT      *dht.DHT
	Identity *notary.Identity
	Backends *StreamBackendRegistry
}

func (s *PeerServer) Ping(ctx context.Context, call schema.Peer_ping) error {
	dhtInstance, err := s.requireDHT()

	if err != nil {
		return err
	}

	sender, err := call.Args().Sender()

	if err != nil {
		return err
	}

	if err := s.rememberSender(sender); err != nil {
		return err
	}

	res, err := call.AllocResults()

	if err != nil {
		return err
	}

	responder, err := res.NewResponder()

	if err != nil {
		return err
	}

	return setNodeInfo(responder, dhtInstance.LocalNode())
}

func (s *PeerServer) FindNode(ctx context.Context, call schema.Peer_findNode) error {
	dhtInstance, err := s.requireDHT()

	if err != nil {
		return err
	}

	sender, err := call.Args().Sender()

	if err != nil {
		return err
	}

	if err := s.rememberSender(sender); err != nil {
		return err
	}

	targetIdData, err := call.Args().TargetId()

	if err != nil {
		return err
	}

	targetId, err := newNodeID(targetIdData)

	if err != nil {
		return err
	}

	closest := dhtInstance.FindClosest(targetId, dht.K)

	res, err := call.AllocResults()

	if err != nil {
		return err
	}

	responder, err := res.NewResponder()

	if err != nil {
		return err
	}

	if err := setNodeInfo(responder, dhtInstance.LocalNode()); err != nil {
		return err
	}

	nodeList, err := res.NewNodes(int32(len(closest)))

	if err != nil {
		return err
	}

	for i, n := range closest {
		info := nodeList.At(i)

		if err := setNodeInfo(info, n); err != nil {
			return err
		}
	}

	return nil
}

func (s *PeerServer) StartCompute(ctx context.Context, call schema.Peer_startCompute) error {
	graphId, err := call.Args().GraphId()

	if err != nil {
		return err
	}

	streamServer := &ComputeStreamServer{
		GraphId:  graphId,
		Identity: s.Identity,
		Backends: s.Backends,
		Profile:  s.DHT.LocalNode().Profile,
	}

	res, err := call.AllocResults()

	if err != nil {
		return err
	}

	stream := schema.ComputeStream_ServerToClient(streamServer)

	if err := res.SetStream(stream); err != nil {
		stream.Release()

		return err
	}

	return nil
}

/*
ComputeStreamServer implements the schema.ComputeStream_Server interface.
It handles streaming data chunks (nodes and tensors) and executing them.
*/
type ComputeStreamServer struct {
	GraphId  string
	Identity *notary.Identity
	Backends *StreamBackendRegistry
	Profile  dht.ComputeProfile
	Nodes    []streamNode
	Tensors  []streamTensor
	mu       sync.Mutex
}

func (s *ComputeStreamServer) WriteNode(ctx context.Context, call schema.ComputeStream_writeNode) error {
	node, err := call.Args().Node()

	if err != nil {
		return err
	}

	ownedNode, err := newStreamNode(node)

	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.Nodes = append(s.Nodes, ownedNode)

	return nil
}

func (s *ComputeStreamServer) WriteTensor(ctx context.Context, call schema.ComputeStream_writeTensor) error {
	tensor, err := call.Args().Tensor()

	if err != nil {
		return err
	}

	ownedTensor, err := newStreamTensor(tensor)

	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.Tensors = append(s.Tensors, ownedTensor)

	return nil
}

func (s *ComputeStreamServer) Execute(ctx context.Context, call schema.ComputeStream_execute) error {
	res, err := call.AllocResults()

	if err != nil {
		return err
	}

	outputs, err := s.execute(ctx)

	if err != nil {
		return setExecutionError(res, err)
	}

	payload, err := resultProofPayload(outputs)

	if err != nil {
		return setExecutionError(res, err)
	}

	if err := res.SetWorkerAddress(s.Identity.Address()); err != nil {
		return err
	}

	if err := res.SetWorkerSignature(s.Identity.Sign(payload)); err != nil {
		return err
	}

	tensorList, err := res.NewOutputs(int32(len(outputs)))

	if err != nil {
		return err
	}

	for index, output := range outputs {
		if err := setSchemaTensor(tensorList.At(index), output); err != nil {
			return err
		}
	}

	return nil
}

func (s *ComputeStreamServer) execute(ctx context.Context) ([]streamTensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if s.Identity == nil {
		return nil, fmt.Errorf("transport: compute stream requires worker identity")
	}

	s.mu.Lock()
	nodes := append([]streamNode(nil), s.Nodes...)
	tensors := append([]streamTensor(nil), s.Tensors...)
	s.mu.Unlock()

	if len(nodes) == 0 {
		return nil, fmt.Errorf("transport: compute stream requires at least one node")
	}

	if s.Backends == nil {
		s.Backends = NewStreamBackendRegistry()
	}

	backend, err := s.Backends.NewForProfile(s.Profile)

	if err != nil {
		return nil, err
	}

	outputs, err := executor.New(backend).Execute(ctx, nodeSpecs(nodes), tensorSpecs(tensors))

	if err != nil {
		return nil, err
	}

	return streamTensorsFromSpecs(outputs), nil
}

func setExecutionError(res schema.ComputeStream_execute_Results, err error) error {
	if err == nil {
		return nil
	}

	return res.SetError(err.Error())
}

func setSchemaTensor(target schema.Tensor, tensor streamTensor) error {
	if err := target.SetId(tensor.ID); err != nil {
		return err
	}

	shapeList, err := target.NewShape(int32(len(tensor.Shape)))

	if err != nil {
		return err
	}

	for index, dimension := range tensor.Shape {
		shapeList.Set(index, dimension)
	}

	if err := target.SetData(tensor.Data); err != nil {
		return err
	}

	return target.SetDtype(tensor.Dtype)
}

func nodeSpecs(nodes []streamNode) []executor.NodeSpec {
	specs := make([]executor.NodeSpec, len(nodes))

	for index, node := range nodes {
		specs[index] = executor.NodeSpec{
			ID:       node.ID,
			Op:       node.Op,
			Inputs:   append([]string(nil), node.Inputs...),
			Metadata: node.Metadata,
			Target:   node.Target,
		}
	}

	return specs
}

func tensorSpecs(tensors []streamTensor) []executor.TensorSpec {
	specs := make([]executor.TensorSpec, len(tensors))

	for index, tensor := range tensors {
		specs[index] = executor.TensorSpec{
			ID:    tensor.ID,
			Shape: append([]int64(nil), tensor.Shape...),
			Data:  append([]byte(nil), tensor.Data...),
			DType: computetensor.DType(tensor.Dtype),
		}
	}

	return specs
}

func streamTensorsFromSpecs(specs []executor.TensorSpec) []streamTensor {
	tensors := make([]streamTensor, len(specs))

	for index, spec := range specs {
		tensors[index] = streamTensor{
			ID:    spec.ID,
			Shape: append([]int64(nil), spec.Shape...),
			Data:  append([]byte(nil), spec.Data...),
			Dtype: string(spec.DType),
		}
	}

	return tensors
}

/*
NodeTransport manages the RPC connection listener and outgoing connections.
*/
type NodeTransport struct {
	localAddr *net.TCPAddr
	listener  *net.TCPListener
	dht       *dht.DHT
	identity  *notary.Identity
	backends  *StreamBackendRegistry
	bootstrap schema.Peer
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
}

func NewNodeTransport(address string, d *dht.DHT) (*NodeTransport, error) {
	if d == nil || d.LocalNode() == nil {
		return nil, fmt.Errorf("transport: DHT with local node is required")
	}

	addr, err := net.ResolveTCPAddr("tcp", address)

	if err != nil {
		return nil, fmt.Errorf("transport: failed to resolve %s: %w", address, err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	identity, err := notary.NewIdentity()

	if err != nil {
		cancel()

		return nil, err
	}

	return &NodeTransport{
		localAddr: addr,
		dht:       d,
		identity:  identity,
		backends:  NewStreamBackendRegistry(),
		ctx:       ctx,
		cancel:    cancel,
	}, nil
}

func (nt *NodeTransport) RegisterStreamBackend(name string, factory StreamBackendFactory) {
	nt.backends.Register(name, factory)
}

func (nt *NodeTransport) Listen() error {
	listener, err := net.ListenTCP("tcp", nt.localAddr)

	if err != nil {
		return fmt.Errorf("transport: failed to listen on %s: %w", nt.localAddr, err)
	}

	nt.listener = listener

	// We update our local node's address since it might have been dynamically assigned port 0
	nt.dht.LocalNode().Address = listener.Addr().String()

	server := &PeerServer{
		DHT:      nt.dht,
		Identity: nt.identity,
		Backends: nt.backends,
	}
	nt.bootstrap = schema.Peer_ServerToClient(server)

	nt.wg.Go(func() {
		for {
			conn, err := nt.listener.AcceptTCP()
			if err != nil {
				if nt.ctx.Err() != nil {
					return
				}

				continue
			}

			// NewConn owns BootstrapClient, so every connection gets its own reference.
			rpcConn := rpc.NewConn(rpc.NewStreamTransport(conn), &rpc.Options{
				BootstrapClient: capnp.Client(nt.bootstrap.AddRef()),
			})

			nt.wg.Go(func() {
				select {
				case <-rpcConn.Done():
				case <-nt.ctx.Done():
					rpcConn.Close()
				}
			})
		}
	})

	return nil
}

// Connect dials a remote node and returns the Bootstrapped Peer capability
func (nt *NodeTransport) Connect(ctx context.Context, address string) (schema.Peer, *rpc.Conn, error) {
	addr, err := net.ResolveTCPAddr("tcp", address)

	if err != nil {
		return schema.Peer{}, nil, err
	}

	conn, err := net.DialTCP("tcp", nil, addr)

	if err != nil {
		return schema.Peer{}, nil, err
	}

	rpcConn := rpc.NewConn(rpc.NewStreamTransport(conn), nil)
	client := schema.Peer(rpcConn.Bootstrap(ctx))

	return client, rpcConn, nil
}

func (nt *NodeTransport) Close() error {
	nt.cancel()

	if nt.listener != nil {
		nt.listener.Close()
	}

	nt.wg.Wait()

	if nt.bootstrap.IsValid() {
		nt.bootstrap.Release()
		nt.bootstrap = schema.Peer{}
	}

	return nil
}

func (nt *NodeTransport) Address() string {
	if nt.listener != nil {
		return nt.listener.Addr().String()
	}

	return nt.localAddr.String()
}

func (s *PeerServer) requireDHT() (*dht.DHT, error) {
	if s == nil || s.DHT == nil || s.DHT.LocalNode() == nil {
		return nil, fmt.Errorf("transport: peer server requires DHT with local node")
	}

	return s.DHT, nil
}

func (s *PeerServer) rememberSender(sender schema.NodeInfo) error {
	idData, err := sender.Id()

	if err != nil {
		return err
	}

	id, err := newNodeID(idData)

	if err != nil {
		return err
	}

	address, err := sender.Address()

	if err != nil {
		return err
	}

	node, err := dht.NewObservedNode(address, id, dht.ComputeProfile{})

	if err != nil {
		return err
	}

	s.DHT.AddNode(node)

	return nil
}

func newNodeID(data []byte) (dht.NodeID, error) {
	var nodeID dht.NodeID

	if len(data) != nodeIDLength {
		return nodeID, fmt.Errorf("transport: node ID must be %d bytes, got %d", nodeIDLength, len(data))
	}

	copy(nodeID[:], data)

	return nodeID, nil
}

func setNodeInfo(info schema.NodeInfo, node *dht.Node) error {
	if node == nil {
		return fmt.Errorf("transport: node info requires node")
	}

	if err := info.SetId(node.ID[:]); err != nil {
		return err
	}

	return info.SetAddress(node.Address)
}

type streamNode struct {
	ID       string
	Op       ir.OpType
	Inputs   []string
	Target   bool
	Config   []byte
	Metadata map[string]any
}

type streamNodeConfig struct {
	Version    int                     `json:"version"`
	Target     bool                    `json:"target"`
	Activation string                  `json:"activation"`
	Operation  string                  `json:"operation"`
	Attributes map[string]ir.Attribute `json:"attributes"`
}

func newStreamNode(node schema.IRNode) (streamNode, error) {
	id, err := node.Id()

	if err != nil {
		return streamNode{}, err
	}

	op, err := node.Op()

	if err != nil {
		return streamNode{}, err
	}

	inputList, err := node.Inputs()

	if err != nil {
		return streamNode{}, err
	}

	inputs := make([]string, inputList.Len())

	for index := range inputs {
		input, err := inputList.At(index)

		if err != nil {
			return streamNode{}, err
		}

		inputs[index] = input
	}

	config, err := node.Config()

	if err != nil {
		return streamNode{}, err
	}

	configData := append([]byte(nil), config...)
	nodeConfig := streamNodeConfig{}
	if len(configData) > 0 {
		if err := json.Unmarshal(configData, &nodeConfig); err != nil {
			return streamNode{}, err
		}
	}

	metadata := map[string]any{
		"activation": nodeConfig.Activation,
	}

	for key, attribute := range nodeConfig.Attributes {
		metadata[key] = attribute.String()
	}

	operation := op
	if nodeConfig.Operation != "" {
		operation = nodeConfig.Operation
	}

	return streamNode{
		ID:       id,
		Op:       normalizeOperation(operation),
		Inputs:   inputs,
		Target:   nodeConfig.Target,
		Config:   configData,
		Metadata: metadata,
	}, nil
}

type streamTensor struct {
	ID    string
	Shape []int64
	Data  []byte
	Dtype string
}

func newStreamTensor(tensor schema.Tensor) (streamTensor, error) {
	id, err := tensor.Id()

	if err != nil {
		return streamTensor{}, err
	}

	shapeList, err := tensor.Shape()

	if err != nil {
		return streamTensor{}, err
	}

	shape := make([]int64, shapeList.Len())

	for index := range shape {
		shape[index] = shapeList.At(index)
	}

	data, err := tensor.Data()

	if err != nil {
		return streamTensor{}, err
	}

	dtype, err := tensor.Dtype()

	if err != nil {
		return streamTensor{}, err
	}

	return streamTensor{
		ID:    id,
		Shape: shape,
		Data:  append([]byte(nil), data...),
		Dtype: dtype,
	}, nil
}

func normalizeOperation(operation string) ir.OpType {
	switch operation {
	case "input", "Input":
		return ir.OpInput
	case "add", "Add":
		return ir.OpAdd
	case "relu", "ReLU":
		return ir.OpReLU
	case "gelu", "GELU":
		return ir.OpGELU
	case "matmul", "Matmul":
		return ir.OpMatmul
	case "fused", "Fused":
		return ir.OpFused
	default:
		return ir.OpType(operation)
	}
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
		return nil, fmt.Errorf("transport: float64 tensor data must be divisible by 8")
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

func ResultProofPayloadFromSchema(outputs schema.Tensor_List) ([]byte, error) {
	tensors := make([]streamTensor, outputs.Len())

	for index := range tensors {
		tensor, err := newStreamTensor(outputs.At(index))

		if err != nil {
			return nil, err
		}

		tensors[index] = tensor
	}

	return resultProofPayload(tensors)
}

func resultProofPayload(outputs []streamTensor) ([]byte, error) {
	buffer := bytes.NewBuffer(nil)

	for _, output := range outputs {
		if err := writeProofString(buffer, output.ID); err != nil {
			return nil, err
		}

		if err := writeProofString(buffer, output.Dtype); err != nil {
			return nil, err
		}

		if err := binary.Write(buffer, binary.LittleEndian, uint64(len(output.Shape))); err != nil {
			return nil, err
		}

		for _, dimension := range output.Shape {
			if err := binary.Write(buffer, binary.LittleEndian, dimension); err != nil {
				return nil, err
			}
		}

		if err := binary.Write(buffer, binary.LittleEndian, uint64(len(output.Data))); err != nil {
			return nil, err
		}

		if _, err := buffer.Write(output.Data); err != nil {
			return nil, err
		}
	}

	return buffer.Bytes(), nil
}

func writeProofString(buffer *bytes.Buffer, value string) error {
	if err := binary.Write(buffer, binary.LittleEndian, uint64(len(value))); err != nil {
		return err
	}

	_, err := buffer.WriteString(value)

	return err
}
