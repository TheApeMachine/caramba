package tools

import (
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tool"
)

/*
MemoryTool provides a unified interface for interacting with multiple memory stores.
*/
type MemoryTool struct {
	buffer   *stream.Buffer
	stores   map[string]io.ReadWriteCloser
	Artifact *tool.Artifact
}

func NewMemoryTool(stores map[string]io.ReadWriteCloser) *MemoryTool {
	errnie.Debug("NewMemoryTool")

	mt := &MemoryTool{
		stores: stores,
		Artifact: tool.New().WithFunction(
			"memory",
			"A tool which can interact with various memory stores through a unified interface.",
			map[string]any{
				"question": map[string]any{
					"type":        "string",
					"description": "A question which is used to retrieve information from a vector database.",
				},
				"keywords": map[string]any{
					"type":        "string",
					"description": "A comma separated list of keywords which are used to retrieve information from all memory stores.",
				},
				"cypher": map[string]any{
					"type":        "string",
					"description": "A Cypher query which is used to retrieve information from a graph database.",
				},
			},
		),
	}

	mt.buffer = stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
		errnie.Debug("MemoryTool.buffer")
		return nil
	})

	return mt
}

func (mt *MemoryTool) Read(p []byte) (n int, err error) {
	errnie.Debug("MemoryTool.Read")
	return mt.buffer.Read(p)
}

func (mt *MemoryTool) Write(p []byte) (n int, err error) {
	errnie.Debug("MemoryTool.Write")
	return mt.buffer.Write(p)
}

func (mt *MemoryTool) Close() error {
	errnie.Debug("MemoryTool.Close")
	return mt.buffer.Close()
}
