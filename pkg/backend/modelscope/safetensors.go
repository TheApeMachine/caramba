package modelscope

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type safetensorsMeta struct {
	DType       string   `json:"dtype,omitempty"`
	Shape       []int64  `json:"shape,omitempty"`
	DataOffsets [2]int64 `json:"data_offsets,omitempty"`
}

/*
parseSafeTensorsReader reads the SafeTensors header from any io.Reader and
builds a GraphData from the tensor metadata. No weight data is read.
*/
func parseSafeTensorsReader(r io.Reader) (GraphData, error) {
	var headerLen uint64
	if err := binary.Read(r, binary.LittleEndian, &headerLen); err != nil {
		return GraphData{}, fmt.Errorf("safetensors: read header length: %w", err)
	}
	if headerLen > 100*1024*1024 {
		return GraphData{}, fmt.Errorf("safetensors: header too large (%d bytes)", headerLen)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return GraphData{}, fmt.Errorf("safetensors: read header: %w", err)
	}

	raw := make(map[string]json.RawMessage)
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return GraphData{}, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}

	b := NewBuilder()
	b.AddNode("__model__", NodeData{"format": "SafeTensors"})

	for name, rawMeta := range raw {
		if name == "__metadata__" {
			var meta map[string]string
			if err := json.Unmarshal(rawMeta, &meta); err == nil {
				data := NodeData{"format": "SafeTensors"}
				for k, v := range meta {
					data[k] = v
				}
				b.AddNode("__model__", data)
			}
			continue
		}

		var meta safetensorsMeta
		if err := json.Unmarshal(rawMeta, &meta); err != nil {
			continue
		}

		shape := make([]string, len(meta.Shape))
		for i, d := range meta.Shape {
			shape[i] = fmt.Sprintf("%d", d)
		}

		b.AddNode(name, NodeData{
			"name":  name,
			"dtype": meta.DType,
			"shape": strings.Join(shape, "×"),
		})

		wireAncestry(b, name)
	}

	return b.Build(), nil
}

/*
wireAncestry walks the full dot-separated ancestry of a tensor name, ensuring
every intermediate node exists and is connected to its parent, up to __model__.
This produces a proper hierarchy tree instead of disconnected leaf pairs.
*/
func wireAncestry(b *Builder, name string) {
	name = strings.ReplaceAll(name, "/", ".")
	parts := strings.Split(name, ".")

	for i := len(parts) - 1; i >= 1; i-- {
		child := strings.Join(parts[:i+1], ".")
		parent := strings.Join(parts[:i], ".")
		b.AddNode(parent, nil)
		b.AddEdge(parent, child, nil)
	}

	// Connect the top-level segment to __model__
	b.AddEdge("__model__", parts[0], nil)
}
