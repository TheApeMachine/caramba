package weights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type tensorMeta struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

/*
IndexFile reads a SafeTensors header from disk.
*/
func IndexFile(path string) (map[string]tensorMeta, int64, error) {
	file, err := os.Open(path)

	if err != nil {
		return nil, 0, err
	}

	defer file.Close()

	return IndexReader(file)
}

/*
IndexReader reads a SafeTensors header from any reader.
*/
func IndexReader(reader io.Reader) (map[string]tensorMeta, int64, error) {
	var headerLength uint64

	if err := binary.Read(reader, binary.LittleEndian, &headerLength); err != nil {
		return nil, 0, fmt.Errorf("safetensors index: %w", err)
	}

	headerBytes := make([]byte, headerLength)

	if _, err := io.ReadFull(reader, headerBytes); err != nil {
		return nil, 0, fmt.Errorf("safetensors index: %w", err)
	}

	raw := make(map[string]json.RawMessage)

	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, 0, fmt.Errorf("safetensors index: %w", err)
	}

	index := make(map[string]tensorMeta, len(raw))

	for name, rawMeta := range raw {
		if name == "__metadata__" {
			continue
		}

		meta := tensorMeta{}

		if err := json.Unmarshal(rawMeta, &meta); err != nil {
			return nil, 0, fmt.Errorf("safetensors tensor %q: %w", name, err)
		}

		index[name] = meta
	}

	return index, int64(headerLength) + 8, nil
}

/*
ReadTensor returns the raw tensor bytes for one entry.
*/
func ReadTensor(path string, tensorName string) ([]byte, tensorMeta, error) {
	index, dataBase, err := IndexFile(path)

	if err != nil {
		return nil, tensorMeta{}, err
	}

	meta, ok := index[tensorName]

	if !ok {
		return nil, tensorMeta{}, fmt.Errorf("safetensors: missing tensor %q", tensorName)
	}

	file, err := os.Open(path)

	if err != nil {
		return nil, tensorMeta{}, err
	}

	defer file.Close()

	start := dataBase + meta.DataOffsets[0]
	length := meta.DataOffsets[1] - meta.DataOffsets[0]
	buffer := make([]byte, length)

	if _, err := file.ReadAt(buffer, start); err != nil {
		return nil, tensorMeta{}, fmt.Errorf("safetensors read %q: %w", tensorName, err)
	}

	return buffer, meta, nil
}
