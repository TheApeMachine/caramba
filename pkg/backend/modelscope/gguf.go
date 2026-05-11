package modelscope

import (
	"encoding/binary"
	"fmt"
	"io"
	"strings"
)

const ggufMagic = 0x46554747 // "GGUF"

type ggufValueType uint32

const (
	ggufUint8   ggufValueType = 0
	ggufInt8    ggufValueType = 1
	ggufUint16  ggufValueType = 2
	ggufInt16   ggufValueType = 3
	ggufUint32  ggufValueType = 4
	ggufInt32   ggufValueType = 5
	ggufFloat32 ggufValueType = 6
	ggufBool    ggufValueType = 7
	ggufString  ggufValueType = 8
	ggufArray   ggufValueType = 9
	ggufUint64  ggufValueType = 10
	ggufInt64   ggufValueType = 11
	ggufFloat64 ggufValueType = 12
)

/*
parseGGUFReader reads the GGUF header from any io.Reader and converts the
result into a GraphData where each tensor is a node and edges connect layers
inferred from naming conventions.
*/
func parseGGUFReader(r io.Reader) (GraphData, error) {
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return GraphData{}, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != ggufMagic {
		return GraphData{}, fmt.Errorf("gguf: not a GGUF file")
	}

	var version uint32
	binary.Read(r, binary.LittleEndian, &version)

	var tensorCount, kvCount uint64
	binary.Read(r, binary.LittleEndian, &tensorCount)
	binary.Read(r, binary.LittleEndian, &kvCount)

	meta := make(map[string]any, kvCount)
	for range kvCount {
		key, err := readGGUFString(r)
		if err != nil {
			break
		}
		val, err := readGGUFValue(r)
		if err != nil {
			break
		}
		meta[key] = val
	}

	type tensorInfo struct {
		name  string
		dims  []uint64
		dtype uint32
	}

	tensors := make([]tensorInfo, 0, tensorCount)
	for range tensorCount {
		name, err := readGGUFString(r)
		if err != nil {
			break
		}
		var nDims uint32
		binary.Read(r, binary.LittleEndian, &nDims)
		dims := make([]uint64, nDims)
		for i := range nDims {
			binary.Read(r, binary.LittleEndian, &dims[i])
		}
		var dtype uint32
		binary.Read(r, binary.LittleEndian, &dtype)
		var offset uint64
		binary.Read(r, binary.LittleEndian, &offset)
		tensors = append(tensors, tensorInfo{name, dims, dtype})
	}

	b := NewBuilder()

	modelMeta := NodeData{"format": "GGUF", "version": version, "tensor_count": tensorCount}
	for k, v := range meta {
		modelMeta[k] = v
	}
	b.AddNode("__model__", modelMeta)

	for _, ti := range tensors {
		shape := make([]string, len(ti.dims))
		for i, d := range ti.dims {
			shape[i] = fmt.Sprintf("%d", d)
		}
		b.AddNode(ti.name, NodeData{
			"name":  ti.name,
			"dtype": ggufDTypeName(ti.dtype),
			"shape": strings.Join(shape, "×"),
		})
		wireAncestry(b, ti.name)
	}

	return b.Build(), nil
}

func ggufDTypeName(t uint32) string {
	names := map[uint32]string{
		0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
		6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
		10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
		14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
		18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
		22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
		26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
	}
	if n, ok := names[t]; ok {
		return n
	}
	return fmt.Sprintf("UNKNOWN(%d)", t)
}

func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readGGUFValue(r io.Reader) (any, error) {
	var vtype ggufValueType
	if err := binary.Read(r, binary.LittleEndian, &vtype); err != nil {
		return nil, err
	}
	return readGGUFTypedValue(r, vtype)
}

func readGGUFTypedValue(r io.Reader, vtype ggufValueType) (any, error) {
	switch vtype {
	case ggufUint8:
		var v uint8
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufInt8:
		var v int8
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufUint16:
		var v uint16
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufInt16:
		var v int16
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufUint32:
		var v uint32
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufInt32:
		var v int32
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufFloat32:
		var v float32
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufBool:
		var v uint8
		binary.Read(r, binary.LittleEndian, &v)
		return v != 0, nil
	case ggufString:
		return readGGUFString(r)
	case ggufArray:
		var elemType ggufValueType
		var count uint64
		binary.Read(r, binary.LittleEndian, &elemType)
		binary.Read(r, binary.LittleEndian, &count)
		for range count {
			readGGUFTypedValue(r, elemType)
		}
		return fmt.Sprintf("[array len=%d]", count), nil
	case ggufUint64:
		var v uint64
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufInt64:
		var v int64
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	case ggufFloat64:
		var v float64
		binary.Read(r, binary.LittleEndian, &v)
		return v, nil
	default:
		return nil, fmt.Errorf("gguf: unknown value type %d", vtype)
	}
}
