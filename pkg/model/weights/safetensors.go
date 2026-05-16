package weights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

const maxSafeTensorsHeader = 100 * 1024 * 1024

var safeTensorMaxInt = int64(int(^uint(0) >> 1))

type safeTensorsFile struct {
	path      string
	dataStart int64
	tensors   map[string]safeTensor
}

type safeTensor struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"-"`
	RawShape    []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

func openSafeTensors(path string) (*safeTensorsFile, error) {
	file, err := os.Open(path)

	if err != nil {
		return nil, fmt.Errorf("safetensors: open %s: %w", path, err)
	}

	defer file.Close()

	var headerLen uint64

	if err := binary.Read(file, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}

	if headerLen > maxSafeTensorsHeader {
		return nil, fmt.Errorf("safetensors: header too large (%d bytes)", headerLen)
	}

	header := make([]byte, headerLen)

	if _, err := io.ReadFull(file, header); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}

	raw := make(map[string]json.RawMessage)

	if err := json.Unmarshal(header, &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parse header: %w", err)
	}

	out := &safeTensorsFile{
		path:      path,
		dataStart: int64(8 + headerLen),
		tensors:   make(map[string]safeTensor, len(raw)),
	}

	for name, encoded := range raw {
		if name == "__metadata__" {
			continue
		}

		var tensor safeTensor

		if err := json.Unmarshal(encoded, &tensor); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}

		shape, elements, err := safeTensorShape(tensor.RawShape)

		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}

		size, err := dtypeSize(tensor.DType)

		if err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}

		expected := int64(elements * size)
		actual := tensor.DataOffsets[1] - tensor.DataOffsets[0]

		if actual != expected {
			return nil, fmt.Errorf(
				"safetensors: tensor %q byte length %d does not match shape*dtype %d",
				name, actual, expected,
			)
		}

		tensor.Shape = shape
		tensor.DType = strings.ToUpper(tensor.DType)
		out.tensors[name] = tensor
	}

	return out, nil
}

func (file *safeTensorsFile) values(tensor safeTensor) ([]float64, error) {
	handle, err := os.Open(file.path)

	if err != nil {
		return nil, fmt.Errorf("safetensors: open %s: %w", file.path, err)
	}

	defer handle.Close()

	offset := file.dataStart + tensor.DataOffsets[0]
	length := tensor.DataOffsets[1] - tensor.DataOffsets[0]
	data := make([]byte, length)

	if _, err := handle.ReadAt(data, offset); err != nil {
		return nil, fmt.Errorf("safetensors: read tensor data: %w", err)
	}

	return decodeTensorValues(tensor.DType, data)
}

func safeTensorShape(raw []int64) ([]int, int, error) {
	shape := make([]int, len(raw))
	elements := 1

	for index, dimension := range raw {
		if dimension < 0 || dimension > safeTensorMaxInt {
			return nil, 0, fmt.Errorf("invalid shape dimension %d", dimension)
		}

		shape[index] = int(dimension)

		if shape[index] == 0 {
			elements = 0
			continue
		}

		if elements > int(safeTensorMaxInt)/shape[index] {
			return nil, 0, fmt.Errorf("shape element count overflows int")
		}

		elements *= shape[index]
	}

	return shape, elements, nil
}

func decodeTensorValues(dtype string, data []byte) ([]float64, error) {
	size, err := dtypeSize(dtype)

	if err != nil {
		return nil, err
	}

	if len(data)%size != 0 {
		return nil, fmt.Errorf("safetensors: %s data length %d is not aligned", dtype, len(data))
	}

	values := make([]float64, len(data)/size)

	switch strings.ToUpper(dtype) {
	case "F64":
		for index := range values {
			bits := binary.LittleEndian.Uint64(data[index*8:])
			values[index] = math.Float64frombits(bits)
		}
	case "F32":
		for index := range values {
			bits := binary.LittleEndian.Uint32(data[index*4:])
			values[index] = float64(math.Float32frombits(bits))
		}
	case "F16":
		for index := range values {
			bits := binary.LittleEndian.Uint16(data[index*2:])
			values[index] = float16ToFloat64(bits)
		}
	case "BF16":
		for index := range values {
			bits := uint32(binary.LittleEndian.Uint16(data[index*2:])) << 16
			values[index] = float64(math.Float32frombits(bits))
		}
	case "I64":
		for index := range values {
			bits := binary.LittleEndian.Uint64(data[index*8:])
			values[index] = float64(int64(bits))
		}
	case "I32":
		for index := range values {
			bits := binary.LittleEndian.Uint32(data[index*4:])
			values[index] = float64(int32(bits))
		}
	case "U32":
		for index := range values {
			values[index] = float64(binary.LittleEndian.Uint32(data[index*4:]))
		}
	case "I16":
		for index := range values {
			values[index] = float64(int16(binary.LittleEndian.Uint16(data[index*2:])))
		}
	case "U16":
		for index := range values {
			values[index] = float64(binary.LittleEndian.Uint16(data[index*2:]))
		}
	case "I8":
		for index := range values {
			values[index] = float64(int8(data[index]))
		}
	case "U8":
		for index := range values {
			values[index] = float64(data[index])
		}
	case "BOOL":
		for index := range values {
			if data[index] != 0 {
				values[index] = 1
			}
		}
	default:
		return nil, fmt.Errorf("safetensors: unsupported dtype %q", dtype)
	}

	return values, nil
}

func dtypeSize(dtype string) (int, error) {
	switch strings.ToUpper(dtype) {
	case "F64", "I64":
		return 8, nil
	case "F32", "I32", "U32":
		return 4, nil
	case "F16", "BF16", "I16", "U16":
		return 2, nil
	case "I8", "U8", "BOOL":
		return 1, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %q", dtype)
	}
}

func float16ToFloat64(bits uint16) float64 {
	sign := 1.0

	if bits&0x8000 != 0 {
		sign = -1
	}

	exponent := int((bits >> 10) & 0x1f)
	fraction := int(bits & 0x03ff)

	switch exponent {
	case 0:
		if fraction == 0 {
			return sign * 0
		}

		return sign * math.Ldexp(float64(fraction), -24)
	case 31:
		if fraction == 0 {
			return sign * math.Inf(1)
		}

		return math.NaN()
	default:
		return sign * math.Ldexp(1+float64(fraction)/1024, exponent-15)
	}
}
