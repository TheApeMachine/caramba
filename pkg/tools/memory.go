package tools

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type MemoryToolData struct {
	Questions []string `json:"questions"`
	Keywords  []string `json:"keywords"`
	Cypher    string   `json:"cypher"`
}

/*
MemoryTool provides a unified interface for interacting with multiple memory stores.
*/
type MemoryTool struct {
	*MemoryToolData
	stores []io.ReadWriteCloser
	enc    *json.Encoder
	dec    *json.Decoder
	in     *bytes.Buffer
	out    *bytes.Buffer
}

func NewMemoryTool(stores ...io.ReadWriteCloser) *MemoryTool {
	errnie.Debug("NewMemoryTool")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	mt := &MemoryTool{
		MemoryToolData: &MemoryToolData{
			Questions: []string{},
			Keywords:  []string{},
			Cypher:    "",
		},
		enc:    json.NewEncoder(out),
		dec:    json.NewDecoder(in),
		in:     in,
		out:    out,
		stores: stores,
	}

	// Pre-encode the tool data to JSON for reading
	mt.enc.Encode(mt.MemoryToolData)

	return mt
}

func (mt *MemoryTool) Read(p []byte) (n int, err error) {
	errnie.Debug("MemoryTool.Read", "p", string(p))

	if mt.out.Len() == 0 {
		return 0, io.EOF
	}

	return mt.out.Read(p)
}

func (mt *MemoryTool) Write(p []byte) (n int, err error) {
	errnie.Debug("MemoryTool.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if mt.out.Len() > 0 {
		mt.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = mt.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf MemoryToolData
	if decErr := mt.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		mt.MemoryToolData.Questions = buf.Questions
		mt.MemoryToolData.Keywords = buf.Keywords
		mt.MemoryToolData.Cypher = buf.Cypher

		// Re-encode to the output buffer for subsequent reads
		if encErr := mt.enc.Encode(mt.MemoryToolData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

func (mt *MemoryTool) Close() error {
	errnie.Debug("MemoryTool.Close")

	for _, store := range mt.stores {
		store.Close()
	}

	return nil
}
