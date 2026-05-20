package runtime

import (
	"context"
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute"
	computeruntime "github.com/theapemachine/caramba/pkg/backend/compute/runtime"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	manifestruntime "github.com/theapemachine/manifesto/runtime"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

/*
GraphBackend executes manifest graphs through the Metal compute stack.
*/
type GraphBackend struct {
	computeBackend *compute.Backend
	deviceID       compute.DeviceID
}

/*
NewGraphBackend constructs a GraphBackend on the default Metal compute device.
*/
func NewGraphBackend(computeBackend *compute.Backend) (*GraphBackend, error) {
	if computeBackend == nil {
		return nil, fmt.Errorf("runtime graph backend: compute backend is required")
	}

	return &GraphBackend{
		computeBackend: computeBackend,
		deviceID:       compute.DeviceID{Location: tensor.Location("metal"), Index: 0},
	}, nil
}

func (backend *GraphBackend) CallGraph(
	ctx context.Context,
	request manifestruntime.GraphCallRequest,
) (manifestruntime.GraphCallResult, error) {
	manifestGraph := request.Graph

	if manifestGraph == nil {
		return manifestruntime.GraphCallResult{}, fmt.Errorf("runtime graph backend: graph is required")
	}

	computeGraph, ok := request.Compute.(*ir.Graph)

	if !ok || computeGraph == nil {
		return manifestruntime.GraphCallResult{}, fmt.Errorf("runtime graph backend: compute graph is required")
	}

	dev, err := backend.computeBackend.Device(backend.deviceID)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	metalDev, ok := dev.(*metal.Backend)
	if !ok {
		return manifestruntime.GraphCallResult{}, fmt.Errorf("runtime graph backend: device is not a metal backend")
	}

	runner := computeruntime.NewMetalGraphRunner(metalDev)

	weightsPath, _ := manifestGraph.Metadata["weights_path"].(string)

	// We need to materialize inputs to tensor.Tensor.
	externalInputs, err := backend.materializeInputs(metalDev, manifestGraph, request.Inputs)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	targets, err := outputTargets(computeGraph, manifestGraph)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	tensors, err := runner.Execute(ctx, computeGraph, targets, weightsPath, externalInputs)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	outputs := make(map[string]any, len(manifestGraph.Outputs))

	for portName, nodeID := range manifestGraph.Outputs {
		value, ok := tensors[nodeID]

		if !ok {
			continue
		}

		native, err := value.Float32Native()

		if err != nil {
			fmt.Printf("Float32Native failed: %v, falling back to RawBytes\n", err)
			// Fallback to downloading raw bytes and converting if Native is unsupported (e.g. Metal)
			outDType, rawBytes, rawErr := value.RawBytes()
			if rawErr != nil {
				return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
			}
			
			// We only support Float32 output for now
			if outDType != dtype.Float32 {
				return manifestruntime.GraphCallResult{}, fmt.Errorf("expected Float32 output, got %s", outDType)
			}
			
			// Convert raw bytes to []float32
			float32s := make([]float32, len(rawBytes)/4)
			for i := range float32s {
				float32s[i] = math.Float32frombits(
					uint32(rawBytes[i*4]) |
					uint32(rawBytes[i*4+1])<<8 |
					uint32(rawBytes[i*4+2])<<16 |
					uint32(rawBytes[i*4+3])<<24,
				)
			}
			native = float32s
		}

		outputs[portName] = native
	}

	return manifestruntime.GraphCallResult{Outputs: outputs}, nil
}

func (backend *GraphBackend) materializeInputs(
	memory *metal.Backend,
	manifestGraph *ast.Graph,
	rawInputs map[string]any,
) (map[string]tensor.Tensor, error) {
	external := make(map[string]tensor.Tensor, len(rawInputs))

	for portName, rawValue := range rawInputs {
		nodeID := portName

		for _, inputName := range manifestGraph.Inputs {
			if inputName == portName {
				nodeID = inputName
			}
		}

		switch typed := rawValue.(type) {
		case []int:
			shape, err := tensor.NewShape([]int{len(typed)})

			if err != nil {
				return nil, err
			}

			tensorValue, err := memory.Upload(shape, dtype.Int32, encodeInt32(typed))

			if err != nil {
				return nil, err
			}

			external[nodeID] = tensorValue
		case []float32:
			shape, err := tensor.NewShape([]int{len(typed)})

			if err != nil {
				return nil, err
			}

			tensorValue, err := memory.Upload(shape, dtype.Float32, float32ToBytes(typed))

			if err != nil {
				return nil, err
			}

			external[nodeID] = tensorValue
		default:
			return nil, fmt.Errorf("runtime graph backend: unsupported input type %T for %q", rawValue, portName)
		}
	}

	return external, nil
}

func outputTargets(computeGraph *ir.Graph, manifestGraph *ast.Graph) ([]*ir.Node, error) {
	index, err := computeGraph.Index()

	if err != nil {
		return nil, err
	}

	targets := make([]*ir.Node, 0, len(manifestGraph.Outputs))

	for _, nodeID := range manifestGraph.Outputs {
		node := index.Node(nodeID)

		if node == nil {
			return nil, fmt.Errorf("runtime graph backend: output node %q is missing", nodeID)
		}

		targets = append(targets, node)
	}

	return targets, nil
}

func encodeInt32(values []int) []byte {
	buffer := make([]byte, len(values)*4)

	for index, element := range values {
		value := uint32(element)
		buffer[index*4] = byte(value)
		buffer[index*4+1] = byte(value >> 8)
		buffer[index*4+2] = byte(value >> 16)
		buffer[index*4+3] = byte(value >> 24)
	}

	return buffer
}

func float32ToBytes(values []float32) []byte {
	buffer := make([]byte, len(values)*4)

	for index, element := range values {
		bits := math.Float32bits(element)
		buffer[index*4] = byte(bits)
		buffer[index*4+1] = byte(bits >> 8)
		buffer[index*4+2] = byte(bits >> 16)
		buffer[index*4+3] = byte(bits >> 24)
	}

	return buffer
}
