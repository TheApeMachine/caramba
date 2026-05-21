package runtime

import (
	"context"
	"encoding/binary"
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
	runner         *computeruntime.MetalGraphRunner
}

/*
NewGraphBackend constructs a GraphBackend on the default Metal compute device.
*/
func NewGraphBackend(computeBackend *compute.Backend) (*GraphBackend, error) {
	if computeBackend == nil {
		return nil, fmt.Errorf("runtime graph backend: compute backend is required")
	}

	dev, err := computeBackend.Device(compute.DeviceID{Location: tensor.Location("metal"), Index: 0})
	if err != nil {
		return nil, err
	}
	metalDev, ok := dev.(*metal.Backend)
	if !ok {
		return nil, fmt.Errorf("runtime graph backend: device is not a metal backend")
	}

	return &GraphBackend{
		computeBackend: computeBackend,
		deviceID:       compute.DeviceID{Location: tensor.Location("metal"), Index: 0},
		runner:         computeruntime.NewMetalGraphRunner(metalDev, dev),
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

	weightsPath, _ := manifestGraph.Metadata["weights_path"].(string)

	// We need to materialize inputs to tensor.Tensor.
	externalInputs, err := backend.materializeInputs(backend.runner.Memory(), manifestGraph, request.Inputs)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	targets, err := outputTargets(computeGraph, manifestGraph)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	tensors, err := backend.runner.Execute(ctx, computeGraph, targets, weightsPath, externalInputs)

	if err != nil {
		return manifestruntime.GraphCallResult{}, err
	}

	outputs := make(map[string]any, len(manifestGraph.Outputs))

	for portName, nodeID := range manifestGraph.Outputs {
		value, ok := tensors[nodeID]

		if !ok {
			continue
		}

		outDType := value.DType()
		var native any

		switch outDType {
		case dtype.Float32:
			f32s, err := value.Float32Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				f32s = make([]float32, len(rawBytes)/4)
				for i := range f32s {
					f32s[i] = math.Float32frombits(binary.LittleEndian.Uint32(rawBytes[i*4:]))
				}
			}
			native = f32s

		case dtype.BFloat16:
			bf16s, err := value.BFloat16Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				bf16s = make([]dtype.BF16, len(rawBytes)/2)
				for i := range bf16s {
					bf16s[i] = dtype.BF16(binary.LittleEndian.Uint16(rawBytes[i*2:]))
				}
			}
			native = bf16s

		case dtype.Float16:
			f16s, err := value.Float16Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				f16s = make([]dtype.F16, len(rawBytes)/2)
				for i := range f16s {
					f16s[i] = dtype.F16(binary.LittleEndian.Uint16(rawBytes[i*2:]))
				}
			}
			native = f16s

		case dtype.Float64:
			f64s, err := value.Float64Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				f64s = make([]float64, len(rawBytes)/8)
				for i := range f64s {
					f64s[i] = math.Float64frombits(binary.LittleEndian.Uint64(rawBytes[i*8:]))
				}
			}
			native = f64s

		case dtype.Int32:
			i32s, err := value.Int32Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				i32s = make([]int32, len(rawBytes)/4)
				for i := range i32s {
					i32s[i] = int32(binary.LittleEndian.Uint32(rawBytes[i*4:]))
				}
			}
			native = i32s

		case dtype.Int64:
			i64s, err := value.Int64Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				i64s = make([]int64, len(rawBytes)/8)
				for i := range i64s {
					i64s[i] = int64(binary.LittleEndian.Uint64(rawBytes[i*8:]))
				}
			}
			native = i64s

		case dtype.Int16:
			i16s, err := value.Int16Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				i16s = make([]int16, len(rawBytes)/2)
				for i := range i16s {
					i16s[i] = int16(binary.LittleEndian.Uint16(rawBytes[i*2:]))
				}
			}
			native = i16s

		case dtype.Int8:
			i8s, err := value.Int8Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				i8s = make([]int8, len(rawBytes))
				for i := range i8s {
					i8s[i] = int8(rawBytes[i])
				}
			}
			native = i8s

		case dtype.Uint64:
			u64s, err := value.Uint64Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				u64s = make([]uint64, len(rawBytes)/8)
				for i := range u64s {
					u64s[i] = binary.LittleEndian.Uint64(rawBytes[i*8:])
				}
			}
			native = u64s

		case dtype.Uint32:
			u32s, err := value.Uint32Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				u32s = make([]uint32, len(rawBytes)/4)
				for i := range u32s {
					u32s[i] = binary.LittleEndian.Uint32(rawBytes[i*4:])
				}
			}
			native = u32s

		case dtype.Uint16:
			u16s, err := value.Uint16Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				u16s = make([]uint16, len(rawBytes)/2)
				for i := range u16s {
					u16s[i] = binary.LittleEndian.Uint16(rawBytes[i*2:])
				}
			}
			native = u16s

		case dtype.Uint8:
			u8s, err := value.Uint8Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				u8s = make([]uint8, len(rawBytes))
				copy(u8s, rawBytes)
			}
			native = u8s

		case dtype.Float8E4M3:
			f8s, err := value.Float8E4M3Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				f8s = make([]dtype.F8E4M3, len(rawBytes))
				for i := range f8s {
					f8s[i] = dtype.F8E4M3(rawBytes[i])
				}
			}
			native = f8s

		case dtype.Float8E5M2:
			f8s, err := value.Float8E5M2Native()
			if err != nil {
				_, rawBytes, rawErr := value.RawBytes()
				if rawErr != nil {
					return manifestruntime.GraphCallResult{}, fmt.Errorf("failed to get raw bytes: %w (original err: %v)", rawErr, err)
				}
				f8s = make([]dtype.F8E5M2, len(rawBytes))
				for i := range f8s {
					f8s[i] = dtype.F8E5M2(rawBytes[i])
				}
			}
			native = f8s

		default:
			return manifestruntime.GraphCallResult{}, fmt.Errorf("runtime graph backend: unsupported output dtype %s", outDType)
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
		case int:
			shape, err := tensor.NewShape([]int{1})
			if err != nil {
				return nil, err
			}
			tensorValue, err := memory.Upload(shape, dtype.Int32, encodeInt32([]int{typed}))
			if err != nil {
				return nil, err
			}
			external[nodeID] = tensorValue
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
		case float32:
			shape, err := tensor.NewShape([]int{1})

			if err != nil {
				return nil, err
			}

			tensorValue, err := memory.Upload(shape, dtype.Float32, float32ToBytes([]float32{typed}))

			if err != nil {
				return nil, err
			}

			external[nodeID] = tensorValue
		case []float32:
			shape, err := tensor.NewShape(float32InputShape(manifestGraph, portName, len(typed)))

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

func float32InputShape(
	manifestGraph *ast.Graph,
	portName string,
	valueCount int,
) []int {
	featureWidth := graphInputFeatureWidth(manifestGraph, portName)

	if featureWidth <= 0 || valueCount%featureWidth != 0 {
		return []int{valueCount}
	}

	return []int{1, valueCount / featureWidth, featureWidth}
}

func graphInputFeatureWidth(manifestGraph *ast.Graph, portName string) int {
	if manifestGraph == nil {
		return 0
	}

	for _, node := range manifestGraph.Nodes {
		if len(node.Inputs) == 0 || node.Inputs[0] != portName {
			continue
		}

		if node.Op != "projection.linear" {
			continue
		}

		return intFromAttribute(node.Attributes["in_features"])
	}

	return 0
}

func intFromAttribute(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
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
